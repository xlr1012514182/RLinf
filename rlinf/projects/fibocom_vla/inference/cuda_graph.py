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

"""Shape-gated PyTorch CUDA Graph capture with exactness evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorSignature:
    """Device, dtype, shape, and stride identity for one tensor."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class ExactnessReport:
    """Comparison between eager and captured outputs."""

    bit_exact: bool
    tensor_count: int
    maximum_absolute_error: float
    maximum_relative_error: float


@dataclass(frozen=True)
class CaptureReport:
    """Capture outcome and path identity."""

    captured: bool
    warmup_iterations: int
    reason: str


def _tree_map(function, value):
    if isinstance(value, Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: _tree_map(function, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(function, item) for item in value)
    if isinstance(value, list):
        return [_tree_map(function, item) for item in value]
    return value


def _tree_zip_apply(function, destination, source) -> None:
    if isinstance(destination, Tensor):
        if not isinstance(source, Tensor):
            raise TypeError("CUDA Graph input tree changed tensor/non-tensor type")
        function(destination, source)
        return
    if isinstance(destination, dict):
        if not isinstance(source, dict) or destination.keys() != source.keys():
            raise TypeError("CUDA Graph input mapping keys changed")
        for key in destination:
            _tree_zip_apply(function, destination[key], source[key])
        return
    if isinstance(destination, (tuple, list)):
        if not isinstance(source, type(destination)) or len(destination) != len(source):
            raise TypeError("CUDA Graph input sequence structure changed")
        for destination_item, source_item in zip(destination, source):
            _tree_zip_apply(function, destination_item, source_item)
        return
    if destination != source:
        raise TypeError("non-tensor CUDA Graph input changed after capture")


def _tensor_signatures(value: Any) -> tuple[TensorSignature, ...]:
    signatures: list[TensorSignature] = []

    def collect(tensor: Tensor) -> Tensor:
        signatures.append(
            TensorSignature(
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        )
        return tensor

    _tree_map(collect, value)
    return tuple(signatures)


def compare_outputs(eager: Any, captured: Any) -> ExactnessReport:
    """Require tree equality and compute floating-point error diagnostics."""

    eager_tensors: list[Tensor] = []
    captured_tensors: list[Tensor] = []

    def collect_eager(tensor: Tensor) -> Tensor:
        eager_tensors.append(tensor.detach())
        return tensor

    def collect_captured(tensor: Tensor) -> Tensor:
        captured_tensors.append(tensor.detach())
        return tensor

    _tree_map(collect_eager, eager)
    _tree_map(collect_captured, captured)
    if len(eager_tensors) != len(captured_tensors):
        raise TypeError(
            "eager and captured output trees contain different tensor counts"
        )
    bit_exact = True
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for eager_tensor, captured_tensor in zip(eager_tensors, captured_tensors):
        if (
            eager_tensor.shape != captured_tensor.shape
            or eager_tensor.dtype != captured_tensor.dtype
        ):
            bit_exact = False
            maximum_absolute = float("inf")
            maximum_relative = float("inf")
            continue
        bit_exact = bit_exact and torch.equal(eager_tensor, captured_tensor)
        if eager_tensor.numel() == 0 or not eager_tensor.is_floating_point():
            continue
        difference = (eager_tensor.float() - captured_tensor.float()).abs()
        maximum_absolute = max(maximum_absolute, float(difference.max().cpu()))
        denominator = eager_tensor.float().abs().clamp_min(1e-12)
        maximum_relative = max(
            maximum_relative, float((difference / denominator).max().cpu())
        )
    return ExactnessReport(
        bit_exact=bit_exact,
        tensor_count=len(eager_tensors),
        maximum_absolute_error=maximum_absolute,
        maximum_relative_error=maximum_relative,
    )


class ShapeStableCUDAGraph:
    """Capture one shape-stable callable and retain eager fallback.

    Only tensor values may change after capture. Shape, stride, dtype, device,
    nested structure, and non-tensor arguments are part of the path identity.
    Dynamic language branches should remain outside this wrapper.
    """

    def __init__(
        self,
        function,
        *,
        warmup_iterations: int = 3,
        eager_fallback: bool = True,
        clone_outputs: bool = True,
    ) -> None:
        if warmup_iterations < 1:
            raise ValueError("warmup_iterations must be positive")
        self.function = function
        self.warmup_iterations = warmup_iterations
        self.eager_fallback = eager_fallback
        self.clone_outputs = clone_outputs
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_args: tuple[Any, ...] | None = None
        self._static_kwargs: dict[str, Any] | None = None
        self._static_output: Any = None
        self._signatures: tuple[TensorSignature, ...] = ()

    @property
    def is_captured(self) -> bool:
        """Whether replay is available."""

        return self._graph is not None

    def capture(self, *args: Any, **kwargs: Any) -> CaptureReport:
        """Warm up on a side stream, then capture the stable region."""

        if not torch.cuda.is_available():
            return CaptureReport(False, 0, "CUDA is unavailable")
        signatures = _tensor_signatures((args, kwargs))
        if not signatures:
            return CaptureReport(False, 0, "no tensor inputs")
        if any(signature.device.type != "cuda" for signature in signatures):
            return CaptureReport(False, 0, "all captured tensor inputs must be CUDA")
        self._static_args = _tree_map(lambda tensor: tensor.detach().clone(), args)
        self._static_kwargs = _tree_map(lambda tensor: tensor.detach().clone(), kwargs)
        self._signatures = _tensor_signatures((self._static_args, self._static_kwargs))
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(self.warmup_iterations):
                self.function(*self._static_args, **self._static_kwargs)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self._graph):
            self._static_output = self.function(
                *self._static_args, **self._static_kwargs
            )
        return CaptureReport(True, self.warmup_iterations, "captured")

    def _matches(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        return _tensor_signatures((args, kwargs)) == self._signatures

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if (
            self._graph is None
            or self._static_args is None
            or self._static_kwargs is None
        ):
            return self.function(*args, **kwargs)
        if not self._matches(args, kwargs):
            if self.eager_fallback:
                return self.function(*args, **kwargs)
            raise ValueError("input signature differs from the captured CUDA Graph")
        _tree_zip_apply(
            lambda destination, source: destination.copy_(source),
            self._static_args,
            args,
        )
        _tree_zip_apply(
            lambda destination, source: destination.copy_(source),
            self._static_kwargs,
            kwargs,
        )
        self._graph.replay()
        if self.clone_outputs:
            return _tree_map(lambda tensor: tensor.clone(), self._static_output)
        return self._static_output

    def verify_exactness(self, *args: Any, **kwargs: Any) -> ExactnessReport:
        """Run paired eager/captured calls and report strict bit equality."""

        if not self.is_captured:
            raise RuntimeError("capture must succeed before exactness verification")
        with torch.inference_mode():
            eager = self.function(*args, **kwargs)
            captured = self(*args, **kwargs)
        return compare_outputs(eager, captured)
