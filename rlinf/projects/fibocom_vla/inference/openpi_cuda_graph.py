# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-closed CUDA Graph boundary for an OpenPI visual-only subgraph.

This module deliberately does not discover a visual encoder by walking an
OpenPI model.  A model-specific adapter must attest to, and expose, one
tensor-only visual capture target.  Tokenization, language processing, and
other data-dependent branches stay in the adapter's eager path.

RTC uses vector-Jacobian products through the denoiser, so RTC, VJP, and any
other autograd request always bypass this executor.  A graph is admitted to
the bounded cache only after its first replay is ``torch.equal`` to the eager
adapter output.  A failed signature is disabled for the executor's lifetime.
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn

from .cuda_graph import CaptureReport, ShapeStableCUDAGraph, compare_outputs


@dataclass(frozen=True)
class OpenPIVisualCaptureContract:
    """Adapter attestation for the exact callable offered for capture."""

    branch_kind: Literal["visual"] = "visual"
    tensor_only: bool = True
    contains_tokenizer: bool = False
    contains_language_model: bool = False
    dynamic_control_flow: bool = False


@runtime_checkable
class OpenPIVisualCaptureTarget(Protocol):
    """One explicit, visual-only module boundary suitable for graph capture."""

    @property
    def name(self) -> str:
        """Stable diagnostic name for this exact capture boundary."""

    @property
    def module(self) -> nn.Module:
        """The module whose eval state and tensor state are validated."""

    @property
    def contract(self) -> OpenPIVisualCaptureContract:
        """Return the adapter's capture-safety attestation."""

    def forward_visual(self, *args: Any, **kwargs: Any) -> Any:
        """Run only the tensor-to-tensor visual region."""


@runtime_checkable
class OpenPIVisualGraphAdapter(Protocol):
    """Model-version-specific bridge around the visual capture boundary.

    ``visual_eager`` is the semantic reference.  It may include eager glue,
    but its returned tensor tree must exactly equal the capture target output
    before that target is admitted to the replay cache.
    """

    def visual_cuda_graph_target(self) -> OpenPIVisualCaptureTarget | None:
        """Return a stable attested target, or ``None`` to require eager.

        The same target instance must be returned for the adapter lifetime;
        target identity is part of the cache signature.
        """

    def visual_eager(self, *args: Any, **kwargs: Any) -> Any:
        """Run the authoritative eager visual path."""


class _RLinfPaliGemmaVisualModule(nn.Module):
    """Exact tensor region used by OpenPI's pinned ``get_image_features``."""

    def __init__(self, vision_tower: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector

    def forward(self, pixel_values: Tensor) -> Tensor:
        image_outputs = self.vision_tower(pixel_values)
        hidden = getattr(image_outputs, "last_hidden_state", None)
        if not isinstance(hidden, Tensor):
            raise TypeError(
                "RLinf OpenPI vision tower must return Tensor last_hidden_state"
            )
        output = self.projector(hidden)
        if not isinstance(output, Tensor):
            raise TypeError("RLinf OpenPI multimodal projector must return a Tensor")
        return output


class RLinfOpenPIVisualCaptureTarget:
    """Concrete capture target for RLinf OpenPI/PaliGemma 4.53.2 layout."""

    def __init__(self, module: _RLinfPaliGemmaVisualModule) -> None:
        self._module = module
        self._contract = OpenPIVisualCaptureContract()

    @property
    def name(self) -> str:
        return "rlinf_openpi.paligemma.vision_tower+multi_modal_projector"

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def contract(self) -> OpenPIVisualCaptureContract:
        return self._contract

    def forward_visual(self, pixel_values: Tensor) -> Tensor:
        return self._module(pixel_values)


class RLinfOpenPIVisualGraphAdapter:
    """Version-checked bridge from RLinf's model to its visual-only region."""

    def __init__(self, model: Any) -> None:
        owner = getattr(model, "paligemma_with_expert", None)
        eager = getattr(owner, "embed_image", None)
        paligemma = getattr(owner, "paligemma", None)
        paligemma_model = getattr(paligemma, "model", None)
        vision_tower = getattr(paligemma_model, "vision_tower", None)
        projector = getattr(paligemma_model, "multi_modal_projector", None)
        if not callable(eager):
            raise TypeError(
                "RLinf OpenPI model lacks paligemma_with_expert.embed_image"
            )
        if not isinstance(vision_tower, nn.Module) or not isinstance(
            projector, nn.Module
        ):
            raise TypeError(
                "RLinf OpenPI model lacks the pinned PaliGemma vision/projector layout"
            )
        owner_module = type(owner).__module__
        if owner_module != "openpi.models_pytorch.gemma_pytorch":
            raise TypeError(
                "unsupported OpenPI visual owner; expected the pinned "
                "openpi.models_pytorch.gemma_pytorch implementation"
            )
        self.model = model
        self.owner = owner
        self._eager = eager
        visual_module = _RLinfPaliGemmaVisualModule(vision_tower, projector)
        visual_module.train(bool(getattr(owner, "training", True)))
        self._target = RLinfOpenPIVisualCaptureTarget(visual_module)

    def visual_cuda_graph_target(self) -> RLinfOpenPIVisualCaptureTarget:
        return self._target

    def visual_eager(self, pixel_values: Tensor) -> Tensor:
        output = self._eager(pixel_values)
        if not isinstance(output, Tensor):
            raise TypeError("RLinf OpenPI embed_image must return a Tensor")
        return output


@dataclass(frozen=True)
class OpenPIVisualExecutionContext:
    """Per-call conditions that force the authoritative eager path."""

    rtc: bool = False
    vjp: bool = False
    requires_autograd: bool = False
    dynamic_branch: bool = False


@dataclass(frozen=True)
class OpenPIVisualGraphDecision:
    """Structured record of the most recent routing decision."""

    path: Literal["eager", "cuda_graph"]
    reason: str
    signature_id: str | None
    target_name: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "path": self.path,
            "reason": self.reason,
            "signature_id": self.signature_id,
            "target_name": self.target_name,
        }


@dataclass(frozen=True)
class OpenPIVisualGraphDiagnostics:
    """Immutable snapshot of cache state, counters, and the last decision."""

    counters: dict[str, int]
    cache_size: int
    cache_capacity: int
    cached_signature_ids: tuple[str, ...]
    disabled_signatures: dict[str, str]
    last_decision: OpenPIVisualGraphDecision | None

    def as_dict(self) -> dict[str, Any]:
        """Return stable diagnostics suitable for policy metadata or logs."""

        return {
            "counters": dict(self.counters),
            "cache_size": self.cache_size,
            "cache_capacity": self.cache_capacity,
            "cached_signature_ids": self.cached_signature_ids,
            "disabled_signatures": dict(self.disabled_signatures),
            "last_decision": (
                None if self.last_decision is None else self.last_decision.as_dict()
            ),
        }


@dataclass(frozen=True)
class _TensorLeafSignature:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _ModuleTensorSignature:
    name: str
    identity: int
    data_pointer: int
    leaf: _TensorLeafSignature


@dataclass(frozen=True)
class _ExecutionSignature:
    target_name: str
    target_identity: int
    module_identity: int
    input_tree: tuple[Any, ...]
    module_state: tuple[_ModuleTensorSignature, ...]
    stream_id: int | None
    thread_token: int | None


@runtime_checkable
class _GraphRunner(Protocol):
    def capture(self, *args: Any, **kwargs: Any) -> CaptureReport:
        """Attempt to capture the callable for one input signature."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Replay the captured callable."""


@dataclass
class _CacheEntry:
    runner: _GraphRunner
    signature_id: str
    target_name: str


class _TensorTreeError(TypeError):
    pass


def _tensor_leaf_signature(tensor: Tensor) -> _TensorLeafSignature:
    return _TensorLeafSignature(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _tensor_tree_signature(value: Any, *, path: str = "root") -> tuple[Any, int]:
    """Return a strict nested signature and tensor count.

    Container keys are structure, not leaves.  Every leaf must be a dense
    strided tensor; values such as prompts, token lists, and ``None`` are
    rejected so a caller cannot accidentally capture the dynamic OpenPI path.
    """

    if isinstance(value, Tensor):
        if value.layout != torch.strided:
            raise _TensorTreeError(f"{path} must be a dense strided tensor")
        return ("tensor", _tensor_leaf_signature(value)), 1
    if isinstance(value, tuple):
        children = [
            _tensor_tree_signature(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return ("tuple", tuple(signature for signature, _ in children)), sum(
            count for _, count in children
        )
    if isinstance(value, list):
        children = [
            _tensor_tree_signature(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return ("list", tuple(signature for signature, _ in children)), sum(
            count for _, count in children
        )
    if isinstance(value, dict):
        children: list[tuple[Any, tuple[Any, int]]] = []
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise _TensorTreeError(
                    f"{path} key {key!r} must be a string or integer"
                )
            children.append(
                (key, _tensor_tree_signature(item, path=f"{path}[{key!r}]"))
            )
        return (
            "dict",
            tuple((key, signature) for key, (signature, _) in children),
        ), sum(count for _, (_, count) in children)
    raise _TensorTreeError(
        f"{path} contains non-tensor leaf {type(value).__name__}; "
        "tokenization, prompts, and dynamic values must remain eager"
    )


def _contains_requires_grad(value: Any) -> bool:
    if isinstance(value, Tensor):
        return value.requires_grad
    if isinstance(value, dict):
        return any(_contains_requires_grad(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_requires_grad(item) for item in value)
    return False


def _clone_tensor_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    raise _TensorTreeError("capture output contains a non-tensor leaf")


def _first_tensor_device(value: Any) -> torch.device | None:
    if isinstance(value, Tensor):
        return value.device
    if isinstance(value, dict):
        for item in value.values():
            device = _first_tensor_device(item)
            if device is not None:
                return device
    elif isinstance(value, (tuple, list)):
        for item in value:
            device = _first_tensor_device(item)
            if device is not None:
                return device
    return None


def _cuda_device_scope(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


def _cuda_stream_id(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    with _cuda_device_scope(device):
        return int(torch.cuda.current_stream(device=device).cuda_stream)


def _module_state_signature(module: nn.Module) -> tuple[_ModuleTensorSignature, ...]:
    state: list[_ModuleTensorSignature] = []
    tensors = list(module.named_parameters(recurse=True))
    tensors.extend(module.named_buffers(recurse=True))
    for name, tensor in tensors:
        state.append(
            _ModuleTensorSignature(
                name=name,
                identity=id(tensor),
                data_pointer=tensor.data_ptr(),
                leaf=_tensor_leaf_signature(tensor),
            )
        )
    return tuple(state)


def _signature_id(signature: _ExecutionSignature) -> str:
    payload = repr(signature).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class OpenPIVisualGraphExecutor:
    """Cache exact, shape-stable OpenPI visual CUDA Graph replays.

    The default backend is :class:`ShapeStableCUDAGraph`.  ``graph_factory``
    exists so the routing and fail-closed logic can be tested without CUDA and
    so a deployment can wrap the same primitive with instrumentation.
    """

    def __init__(
        self,
        adapter: OpenPIVisualGraphAdapter,
        *,
        cache_capacity: int = 4,
        warmup_iterations: int = 3,
        graph_factory: Callable[[Callable[..., Any]], _GraphRunner] | None = None,
    ) -> None:
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be positive")
        if warmup_iterations < 1:
            raise ValueError("warmup_iterations must be positive")
        if not isinstance(adapter, OpenPIVisualGraphAdapter):
            raise TypeError("adapter must implement OpenPIVisualGraphAdapter")
        self.adapter = adapter
        self.cache_capacity = cache_capacity
        self.warmup_iterations = warmup_iterations
        self._graph_factory = graph_factory or self._default_graph_factory
        self._cache: OrderedDict[_ExecutionSignature, _CacheEntry] = OrderedDict()
        self._disabled: dict[_ExecutionSignature, str] = {}
        self._counters: Counter[str] = Counter()
        self._last_decision: OpenPIVisualGraphDecision | None = None
        self._thread_tokens: WeakKeyDictionary[threading.Thread, int] = (
            WeakKeyDictionary()
        )
        self._next_thread_token = 0
        # Cache routing and runner dispatch are serialized.  Since a CUDA
        # stream and its host thread are part of each CUDA signature, distinct
        # streams and per-thread default streams never share a runner's static
        # buffers.  Same-signature copies/replays/clones remain CUDA-ordered
        # after this host-side lock is released.
        self._routing_lock = threading.RLock()

    def _current_thread_token(self) -> int:
        thread = threading.current_thread()
        token = self._thread_tokens.get(thread)
        if token is None:
            self._next_thread_token += 1
            token = self._next_thread_token
            self._thread_tokens[thread] = token
        return token

    def _default_graph_factory(self, function: Callable[..., Any]) -> _GraphRunner:
        return ShapeStableCUDAGraph(
            function,
            warmup_iterations=self.warmup_iterations,
            eager_fallback=False,
            clone_outputs=True,
        )

    def _decide(
        self,
        *,
        path: Literal["eager", "cuda_graph"],
        reason: str,
        signature_id: str | None,
        target_name: str | None,
    ) -> None:
        self._last_decision = OpenPIVisualGraphDecision(
            path=path,
            reason=reason,
            signature_id=signature_id,
            target_name=target_name,
        )

    def _eager(
        self,
        *args: Any,
        preserve_autograd: bool,
        **kwargs: Any,
    ) -> Any:
        self._counters["eager_calls"] += 1
        if preserve_autograd:
            return self.adapter.visual_eager(*args, **kwargs)
        with torch.inference_mode():
            return self.adapter.visual_eager(*args, **kwargs)

    def _disable(self, signature: _ExecutionSignature, reason: str) -> None:
        self._cache.pop(signature, None)
        self._disabled.setdefault(signature, reason)

    @staticmethod
    def _contract_rejection(target: OpenPIVisualCaptureTarget) -> str | None:
        contract = target.contract
        if contract.branch_kind != "visual":
            return "capture target is not an attested visual branch"
        if not contract.tensor_only:
            return "capture target is not tensor-only"
        if contract.contains_tokenizer:
            return "tokenizer must remain on the eager path"
        if contract.contains_language_model:
            return "language model must remain on the eager path"
        if contract.dynamic_control_flow:
            return "dynamic control flow must remain on the eager path"
        return None

    @staticmethod
    def _runtime_bypasses(
        context: OpenPIVisualExecutionContext,
        input_tree: Any,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if context.rtc:
            reasons.append("rtc_request")
        if context.vjp:
            reasons.append("vjp_request")
        if context.requires_autograd:
            reasons.append("autograd_request")
        if context.dynamic_branch:
            reasons.append("dynamic_branch_request")
        if _contains_requires_grad(input_tree):
            reasons.append("input_requires_grad")
        return tuple(reasons)

    @staticmethod
    def _validate_devices(
        input_tree: Any,
        module_state: tuple[_ModuleTensorSignature, ...],
    ) -> str | None:
        input_devices: set[torch.device] = set()

        def collect(value: Any) -> None:
            if isinstance(value, Tensor):
                input_devices.add(value.device)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(input_tree)
        if len(input_devices) != 1:
            return "capture inputs must use exactly one fixed device"
        module_devices = {item.leaf.device for item in module_state}
        if len(module_devices) > 1:
            return "capture module state spans multiple devices"
        if module_devices and module_devices != input_devices:
            return "capture module state and inputs use different devices"
        return None

    def __call__(
        self,
        *args: Any,
        graph_context: OpenPIVisualExecutionContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the visual branch through a verified graph or eager fallback."""

        with self._routing_lock:
            return self._call_serialized(
                *args,
                graph_context=graph_context,
                **kwargs,
            )

    def _call_serialized(
        self,
        *args: Any,
        graph_context: OpenPIVisualExecutionContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route one call while cache and static-buffer dispatch are locked."""

        self._counters["calls"] += 1
        context = graph_context or OpenPIVisualExecutionContext()
        input_tree = (args, kwargs)
        bypasses = self._runtime_bypasses(context, input_tree)
        if bypasses:
            for reason in bypasses:
                self._counters[f"bypass_{reason}"] += 1
            self._decide(
                path="eager",
                reason="+".join(bypasses),
                signature_id=None,
                target_name=None,
            )
            return self._eager(
                *args,
                preserve_autograd=any(
                    reason in {"rtc_request", "vjp_request", "autograd_request"}
                    for reason in bypasses
                )
                or "input_requires_grad" in bypasses,
                **kwargs,
            )

        target = self.adapter.visual_cuda_graph_target()
        if target is None:
            self._counters["capture_rejections"] += 1
            self._decide(
                path="eager",
                reason="adapter did not expose a visual capture target",
                signature_id=None,
                target_name=None,
            )
            return self._eager(*args, preserve_autograd=False, **kwargs)
        if not isinstance(target, OpenPIVisualCaptureTarget):
            raise TypeError("visual target must implement OpenPIVisualCaptureTarget")
        if not target.name.strip():
            raise ValueError("visual capture target name must not be blank")
        if not isinstance(target.module, nn.Module):
            raise TypeError("visual capture target module must be torch.nn.Module")
        if target.module.training:
            self._counters["capture_rejections"] += 1
            self._counters["bypass_module_training"] += 1
            self._decide(
                path="eager",
                reason="visual capture module must be in eval mode",
                signature_id=None,
                target_name=target.name,
            )
            return self._eager(*args, preserve_autograd=False, **kwargs)
        contract_rejection = self._contract_rejection(target)
        if contract_rejection is not None:
            self._counters["capture_rejections"] += 1
            self._counters["bypass_contract"] += 1
            self._decide(
                path="eager",
                reason=contract_rejection,
                signature_id=None,
                target_name=target.name,
            )
            return self._eager(*args, preserve_autograd=False, **kwargs)

        try:
            tree_signature, tensor_count = _tensor_tree_signature(input_tree)
        except _TensorTreeError as error:
            self._counters["capture_rejections"] += 1
            self._counters["bypass_non_tensor_input"] += 1
            self._decide(
                path="eager",
                reason=str(error),
                signature_id=None,
                target_name=target.name,
            )
            return self._eager(*args, preserve_autograd=False, **kwargs)
        if tensor_count == 0:
            self._counters["capture_rejections"] += 1
            self._counters["bypass_no_tensor_input"] += 1
            self._decide(
                path="eager",
                reason="visual capture target received no tensor inputs",
                signature_id=None,
                target_name=target.name,
            )
            return self._eager(*args, preserve_autograd=False, **kwargs)

        module_state = _module_state_signature(target.module)
        device_rejection = self._validate_devices(input_tree, module_state)
        if device_rejection is not None:
            self._counters["capture_rejections"] += 1
            self._counters["bypass_device_contract"] += 1
            self._decide(
                path="eager",
                reason=device_rejection,
                signature_id=None,
                target_name=target.name,
            )
            return self._eager(*args, preserve_autograd=False, **kwargs)

        execution_device = _first_tensor_device(input_tree)
        if execution_device is None:
            raise RuntimeError("validated visual input tree unexpectedly has no tensor")
        stream_id = _cuda_stream_id(execution_device)
        thread_token = (
            self._current_thread_token() if execution_device.type == "cuda" else None
        )

        signature = _ExecutionSignature(
            target_name=target.name,
            target_identity=id(target),
            module_identity=id(target.module),
            input_tree=tree_signature,
            module_state=module_state,
            stream_id=stream_id,
            thread_token=thread_token,
        )
        current_signature_id = _signature_id(signature)
        disabled_reason = self._disabled.get(signature)
        if disabled_reason is not None:
            self._counters["disabled_signature_calls"] += 1
            self._decide(
                path="eager",
                reason=f"signature disabled: {disabled_reason}",
                signature_id=current_signature_id,
                target_name=target.name,
            )
            with _cuda_device_scope(execution_device):
                return self._eager(*args, preserve_autograd=False, **kwargs)

        entry = self._cache.get(signature)
        if entry is not None:
            self._cache.move_to_end(signature)
            try:
                with _cuda_device_scope(execution_device), torch.inference_mode():
                    output = entry.runner(*args, **kwargs)
            except Exception as error:
                reason = f"graph replay failed: {type(error).__name__}: {error}"
                self._counters["replay_failures"] += 1
                self._disable(signature, reason)
                self._decide(
                    path="eager",
                    reason=reason,
                    signature_id=current_signature_id,
                    target_name=target.name,
                )
                with _cuda_device_scope(execution_device):
                    return self._eager(*args, preserve_autograd=False, **kwargs)
            self._counters["graph_replays"] += 1
            self._decide(
                path="cuda_graph",
                reason="verified signature cache hit",
                signature_id=current_signature_id,
                target_name=target.name,
            )
            return output

        self._counters["signature_misses"] += 1
        with _cuda_device_scope(execution_device):
            eager_output = self._eager(*args, preserve_autograd=False, **kwargs)
        try:
            eager_output_signature, output_tensor_count = _tensor_tree_signature(
                eager_output, path="output"
            )
            if output_tensor_count == 0:
                raise _TensorTreeError("capture output contains no tensors")
            eager_reference = _clone_tensor_tree(eager_output)
        except _TensorTreeError as error:
            reason = str(error)
            self._counters["capture_rejections"] += 1
            self._counters["bypass_non_tensor_output"] += 1
            self._disable(signature, reason)
            self._decide(
                path="eager",
                reason=reason,
                signature_id=current_signature_id,
                target_name=target.name,
            )
            return eager_output

        self._counters["capture_attempts"] += 1
        try:
            with _cuda_device_scope(execution_device):
                runner = self._graph_factory(target.forward_visual)
                capture_report = runner.capture(*args, **kwargs)
        except Exception as error:
            reason = f"graph capture failed: {type(error).__name__}: {error}"
            self._counters["capture_rejections"] += 1
            self._disable(signature, reason)
            self._decide(
                path="eager",
                reason=reason,
                signature_id=current_signature_id,
                target_name=target.name,
            )
            return eager_output
        if not capture_report.captured:
            reason = f"graph capture rejected: {capture_report.reason}"
            self._counters["capture_rejections"] += 1
            self._disable(signature, reason)
            self._decide(
                path="eager",
                reason=reason,
                signature_id=current_signature_id,
                target_name=target.name,
            )
            return eager_output

        try:
            with _cuda_device_scope(execution_device), torch.inference_mode():
                captured_output = runner(*args, **kwargs)
            captured_output_signature, _ = _tensor_tree_signature(
                captured_output, path="captured_output"
            )
            if captured_output_signature != eager_output_signature:
                raise _TensorTreeError(
                    "captured output structure, shape, stride, dtype, or device "
                    "differs from eager"
                )
            exactness = compare_outputs(eager_reference, captured_output)
        except Exception as error:
            reason = f"first replay comparison failed: {type(error).__name__}: {error}"
            self._counters["parity_failures"] += 1
            self._disable(signature, reason)
            self._decide(
                path="eager",
                reason=reason,
                signature_id=current_signature_id,
                target_name=target.name,
            )
            return eager_output
        if not exactness.bit_exact:
            reason = (
                "first replay is not bit-exact to eager "
                f"(max_abs={exactness.maximum_absolute_error}, "
                f"max_rel={exactness.maximum_relative_error})"
            )
            self._counters["parity_failures"] += 1
            self._disable(signature, reason)
            self._decide(
                path="eager",
                reason=reason,
                signature_id=current_signature_id,
                target_name=target.name,
            )
            return eager_output

        if len(self._cache) >= self.cache_capacity:
            self._cache.popitem(last=False)
            self._counters["cache_evictions"] += 1
        self._cache[signature] = _CacheEntry(
            runner=runner,
            signature_id=current_signature_id,
            target_name=target.name,
        )
        self._counters["captures_succeeded"] += 1
        self._decide(
            path="eager",
            reason="first call returned eager after bit-exact capture verification",
            signature_id=current_signature_id,
            target_name=target.name,
        )
        return eager_output

    def diagnostics(self) -> OpenPIVisualGraphDiagnostics:
        """Return a detached, structured snapshot of executor state."""

        with self._routing_lock:
            return OpenPIVisualGraphDiagnostics(
                counters=dict(sorted(self._counters.items())),
                cache_size=len(self._cache),
                cache_capacity=self.cache_capacity,
                cached_signature_ids=tuple(
                    entry.signature_id for entry in self._cache.values()
                ),
                disabled_signatures={
                    _signature_id(signature): reason
                    for signature, reason in self._disabled.items()
                },
                last_decision=self._last_decision,
            )


class InstalledRLinfOpenPIVisualGraph:
    """Reversible patch routing the real OpenPI ``embed_image`` call site."""

    def __init__(
        self,
        adapter: RLinfOpenPIVisualGraphAdapter,
        executor: OpenPIVisualGraphExecutor,
        *,
        default_context: OpenPIVisualExecutionContext | None = None,
    ) -> None:
        if executor.adapter is not adapter:
            raise ValueError("executor must own the exact adapter being installed")
        self.adapter = adapter
        self.executor = executor
        self.default_context = default_context or OpenPIVisualExecutionContext()
        self._local = threading.local()
        self._installed = False
        self._original = adapter.owner.embed_image

    def _current_context(self) -> OpenPIVisualExecutionContext:
        stack = getattr(self._local, "contexts", ())
        return stack[-1] if stack else self.default_context

    def __call__(self, pixel_values: Tensor) -> Tensor:
        output = self.executor(
            pixel_values,
            graph_context=self._current_context(),
        )
        if not isinstance(output, Tensor):
            raise TypeError("installed RLinf OpenPI visual graph must return a Tensor")
        return output

    def install(self) -> "InstalledRLinfOpenPIVisualGraph":
        if self._installed:
            return self
        existing = getattr(self.adapter.owner, "_fibocom_visual_graph_patch", None)
        if existing is not None and existing is not self:
            raise RuntimeError("RLinf OpenPI visual graph is already patched")
        self.adapter.owner.embed_image = self
        self.adapter.owner._fibocom_visual_graph_patch = self
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        if getattr(self.adapter.owner, "embed_image", None) is not self:
            raise RuntimeError("RLinf OpenPI embed_image changed after graph install")
        self.adapter.owner.embed_image = self._original
        delattr(self.adapter.owner, "_fibocom_visual_graph_patch")
        self._installed = False

    @contextmanager
    def execution_context(self, context: OpenPIVisualExecutionContext):
        """Temporarily force RTC/VJP/dynamic calls onto the eager path."""

        if not isinstance(context, OpenPIVisualExecutionContext):
            raise TypeError("context must be an OpenPIVisualExecutionContext")
        stack = getattr(self._local, "contexts", ())
        self._local.contexts = (*stack, context)
        try:
            yield self
        finally:
            current = getattr(self._local, "contexts", ())
            if not current or current[-1] is not context:
                raise RuntimeError(
                    "OpenPI visual execution-context stack was corrupted"
                )
            self._local.contexts = current[:-1]


def install_rlinf_openpi_visual_graph(
    model: Any,
    *,
    cache_capacity: int = 4,
    warmup_iterations: int = 3,
    default_context: OpenPIVisualExecutionContext | None = None,
) -> InstalledRLinfOpenPIVisualGraph:
    """Install the concrete visual-only executor at RLinf's real call site."""

    adapter = RLinfOpenPIVisualGraphAdapter(model)
    executor = OpenPIVisualGraphExecutor(
        adapter,
        cache_capacity=cache_capacity,
        warmup_iterations=warmup_iterations,
    )
    return InstalledRLinfOpenPIVisualGraph(
        adapter,
        executor,
        default_context=default_context,
    ).install()
