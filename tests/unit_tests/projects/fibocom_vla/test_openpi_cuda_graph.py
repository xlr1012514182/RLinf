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

"""CPU routing tests and an optional real-CUDA smoke for the visual graph."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from rlinf.projects.fibocom_vla.inference.cuda_graph import (  # noqa: E402
    CaptureReport,
)
from rlinf.projects.fibocom_vla.inference.openpi_cuda_graph import (  # noqa: E402
    InstalledRLinfOpenPIVisualGraph,
    OpenPIVisualCaptureContract,
    OpenPIVisualExecutionContext,
    OpenPIVisualGraphExecutor,
    RLinfOpenPIVisualGraphAdapter,
)


class _Scale(torch.nn.Module):
    def forward(self, value: torch.Tensor, ignored: Any = None) -> torch.Tensor:
        del ignored
        return value * 2


class _Target:
    def __init__(
        self,
        module: torch.nn.Module,
        contract: OpenPIVisualCaptureContract | None = None,
    ) -> None:
        self._module = module
        self._contract = contract or OpenPIVisualCaptureContract()

    @property
    def name(self) -> str:
        return "fake.visual_encoder"

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    @property
    def contract(self) -> OpenPIVisualCaptureContract:
        return self._contract

    def forward_visual(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


class _Adapter:
    def __init__(
        self,
        target: _Target | None,
        *,
        eager_transform=None,
    ) -> None:
        self.target = target
        self.eager_transform = eager_transform
        self.eager_calls = 0

    def visual_cuda_graph_target(self) -> _Target | None:
        return self.target

    def visual_eager(self, *args: Any, **kwargs: Any) -> Any:
        self.eager_calls += 1
        if self.target is None:
            return args[0] * 2
        output = self.target.forward_visual(*args, **kwargs)
        if self.eager_transform is not None:
            return self.eager_transform(output)
        return output


class _FakeRunner:
    def __init__(
        self,
        function,
        *,
        corrupt_values: bool = False,
        corrupt_structure: bool = False,
    ) -> None:
        self.function = function
        self.corrupt_values = corrupt_values
        self.corrupt_structure = corrupt_structure
        self.capture_calls = 0
        self.replay_calls = 0

    def capture(self, *args: Any, **kwargs: Any) -> CaptureReport:
        del args, kwargs
        self.capture_calls += 1
        return CaptureReport(captured=True, warmup_iterations=1, reason="fake")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.replay_calls += 1
        output = self.function(*args, **kwargs)
        if self.corrupt_values:
            output = output + 1
        if self.corrupt_structure:
            output = (output,)
        return output


class _FakeFactory:
    def __init__(
        self,
        *,
        corrupt_values: bool = False,
        corrupt_structure: bool = False,
    ) -> None:
        self.corrupt_values = corrupt_values
        self.corrupt_structure = corrupt_structure
        self.runners: list[_FakeRunner] = []

    def __call__(self, function) -> _FakeRunner:
        runner = _FakeRunner(
            function,
            corrupt_values=self.corrupt_values,
            corrupt_structure=self.corrupt_structure,
        )
        self.runners.append(runner)
        return runner


class _BlockingRunner(_FakeRunner):
    def __init__(self, function, tracker: dict[str, Any]) -> None:
        super().__init__(function)
        self.tracker = tracker

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.tracker["lock"]:
            self.tracker["active"] += 1
            self.tracker["maximum"] = max(
                self.tracker["maximum"], self.tracker["active"]
            )
        try:
            time.sleep(0.02)
            return super().__call__(*args, **kwargs)
        finally:
            with self.tracker["lock"]:
                self.tracker["active"] -= 1


class _BlockingFactory(_FakeFactory):
    def __init__(self) -> None:
        super().__init__()
        self.tracker = {"lock": threading.Lock(), "active": 0, "maximum": 0}

    def __call__(self, function) -> _BlockingRunner:
        runner = _BlockingRunner(function, self.tracker)
        self.runners.append(runner)
        return runner


def _executor(
    *,
    contract: OpenPIVisualCaptureContract | None = None,
    factory: _FakeFactory | None = None,
    cache_capacity: int = 4,
) -> tuple[OpenPIVisualGraphExecutor, _Adapter, _FakeFactory]:
    module = _Scale().eval()
    adapter = _Adapter(_Target(module, contract))
    resolved_factory = factory or _FakeFactory()
    executor = OpenPIVisualGraphExecutor(
        adapter,
        cache_capacity=cache_capacity,
        graph_factory=resolved_factory,
    )
    return executor, adapter, resolved_factory


def test_training_dynamic_and_non_tensor_paths_are_never_captured() -> None:
    training_module = _Scale().train()
    training_adapter = _Adapter(_Target(training_module))
    training_factory = _FakeFactory()
    training = OpenPIVisualGraphExecutor(
        training_adapter,
        graph_factory=training_factory,
    )

    torch.testing.assert_close(training(torch.ones(2)), torch.full((2,), 2.0))
    assert not training_factory.runners
    assert "eval mode" in training.diagnostics().last_decision.reason

    dynamic_contract = OpenPIVisualCaptureContract(dynamic_control_flow=True)
    dynamic, _, dynamic_factory = _executor(contract=dynamic_contract)
    torch.testing.assert_close(dynamic(torch.ones(2)), torch.full((2,), 2.0))
    assert not dynamic_factory.runners
    assert "dynamic control flow" in dynamic.diagnostics().last_decision.reason

    non_tensor, _, non_tensor_factory = _executor()
    torch.testing.assert_close(
        non_tensor(torch.ones(2), "prompt stays eager"),
        torch.full((2,), 2.0),
    )
    diagnostics = non_tensor.diagnostics()
    assert not non_tensor_factory.runners
    assert diagnostics.counters["bypass_non_tensor_input"] == 1
    assert "non-tensor leaf str" in diagnostics.last_decision.reason


@pytest.mark.parametrize(
    ("contract", "expected_reason"),
    [
        (OpenPIVisualCaptureContract(tensor_only=False), "not tensor-only"),
        (OpenPIVisualCaptureContract(contains_tokenizer=True), "tokenizer"),
        (OpenPIVisualCaptureContract(contains_language_model=True), "language model"),
    ],
)
def test_language_and_tokenizer_contracts_stay_eager(
    contract: OpenPIVisualCaptureContract,
    expected_reason: str,
) -> None:
    executor, _, factory = _executor(contract=contract)

    executor(torch.ones(1))

    assert not factory.runners
    assert expected_reason in executor.diagnostics().last_decision.reason


def test_shape_and_stride_are_distinct_signatures_in_a_bounded_lru() -> None:
    executor, _, factory = _executor(cache_capacity=1)
    contiguous = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    transposed = contiguous.t()
    assert contiguous.shape == transposed.shape
    assert contiguous.stride() != transposed.stride()

    first = executor(contiguous)
    first_replay = executor(contiguous + 1)
    second = executor(transposed)
    second_replay = executor((transposed + 1).t().contiguous().t())

    torch.testing.assert_close(first, contiguous * 2)
    torch.testing.assert_close(first_replay, (contiguous + 1) * 2)
    torch.testing.assert_close(second, transposed * 2)
    torch.testing.assert_close(
        second_replay, ((transposed + 1).t().contiguous().t()) * 2
    )
    assert len(factory.runners) == 2
    diagnostics = executor.diagnostics()
    assert diagnostics.counters["captures_succeeded"] == 2
    assert diagnostics.counters["graph_replays"] == 2
    assert diagnostics.counters["cache_evictions"] == 1
    assert diagnostics.cache_size == 1
    assert diagnostics.last_decision.path == "cuda_graph"


def test_rtc_vjp_autograd_and_dynamic_requests_bypass_cached_graph() -> None:
    executor, _, factory = _executor()
    value = torch.ones(2)
    executor(value)
    executor(value)
    assert factory.runners[0].replay_calls == 2  # parity replay + one cache hit

    for context in (
        OpenPIVisualExecutionContext(rtc=True),
        OpenPIVisualExecutionContext(vjp=True),
        OpenPIVisualExecutionContext(requires_autograd=True),
        OpenPIVisualExecutionContext(dynamic_branch=True),
    ):
        torch.testing.assert_close(
            executor(value, graph_context=context),
            value * 2,
        )

    grad_value = torch.ones(2, requires_grad=True)
    grad_output = executor(grad_value)
    assert grad_output.requires_grad
    grad_output.sum().backward()
    torch.testing.assert_close(grad_value.grad, torch.full((2,), 2.0))

    assert factory.runners[0].replay_calls == 2
    counters = executor.diagnostics().counters
    assert counters["bypass_rtc_request"] == 1
    assert counters["bypass_vjp_request"] == 1
    assert counters["bypass_autograd_request"] == 1
    assert counters["bypass_dynamic_branch_request"] == 1
    assert counters["bypass_input_requires_grad"] == 1


@pytest.mark.parametrize("corrupt_structure", [False, True])
def test_first_replay_parity_failure_permanently_disables_signature(
    corrupt_structure: bool,
) -> None:
    factory = _FakeFactory(
        corrupt_values=not corrupt_structure,
        corrupt_structure=corrupt_structure,
    )
    executor, _, _ = _executor(factory=factory)
    value = torch.tensor([1.0, 3.0])

    first = executor(value)
    second = executor(value + 1)

    torch.testing.assert_close(first, value * 2)
    torch.testing.assert_close(second, (value + 1) * 2)
    assert len(factory.runners) == 1
    assert factory.runners[0].capture_calls == 1
    assert factory.runners[0].replay_calls == 1
    diagnostics = executor.diagnostics()
    assert diagnostics.counters["capture_attempts"] == 1
    assert diagnostics.counters["parity_failures"] == 1
    assert diagnostics.counters["disabled_signature_calls"] == 1
    assert diagnostics.cache_size == 0
    assert len(diagnostics.disabled_signatures) == 1
    assert diagnostics.last_decision.reason.startswith("signature disabled:")


def test_default_shape_stable_backend_rejects_cpu_once_then_stays_eager() -> None:
    module = _Scale().eval()
    adapter = _Adapter(_Target(module))
    executor = OpenPIVisualGraphExecutor(adapter, warmup_iterations=1)
    value = torch.ones(2)

    torch.testing.assert_close(executor(value), value * 2)
    torch.testing.assert_close(executor(value + 1), (value + 1) * 2)

    diagnostics = executor.diagnostics()
    assert diagnostics.counters["capture_attempts"] == 1
    assert diagnostics.counters["capture_rejections"] == 1
    assert diagnostics.counters["disabled_signature_calls"] == 1
    assert len(diagnostics.disabled_signatures) == 1


def test_concurrent_replays_do_not_overlap_one_runner_static_buffer() -> None:
    factory = _BlockingFactory()
    executor, _, _ = _executor(factory=factory)
    executor(torch.ones(2))
    start = threading.Barrier(5)
    results: list[tuple[int, torch.Tensor]] = []
    errors: list[BaseException] = []

    def replay(index: int) -> None:
        try:
            start.wait()
            value = torch.full((2,), float(index))
            results.append((index, executor(value)))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=replay, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert not errors
    assert factory.tracker["maximum"] == 1
    assert len(factory.runners) == 1
    for index, output in results:
        torch.testing.assert_close(output, torch.full((2,), float(index * 2)))
    diagnostics = executor.diagnostics()
    assert diagnostics.counters["graph_replays"] == 4
    assert diagnostics.cache_size == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_distinct_cuda_streams_use_distinct_static_buffer_runners() -> None:
    factory = _FakeFactory()
    module = _Scale().cuda().eval()
    adapter = _Adapter(_Target(module))
    executor = OpenPIVisualGraphExecutor(adapter, graph_factory=factory)
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    value = torch.ones(2, device="cuda")

    with torch.cuda.stream(stream_a):
        first_a = executor(value)
    with torch.cuda.stream(stream_b):
        first_b = executor(value + 1)
    with torch.cuda.stream(stream_a):
        replay_a = executor(value + 2)
    torch.cuda.synchronize()

    assert len(factory.runners) == 2
    assert executor.diagnostics().cache_size == 2
    assert torch.equal(first_a, value * 2)
    assert torch.equal(first_b, (value + 1) * 2)
    assert torch.equal(replay_a, (value + 2) * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_default_stream_runner_is_not_shared_across_threads() -> None:
    factory = _FakeFactory()
    module = _Scale().cuda().eval()
    executor = OpenPIVisualGraphExecutor(
        _Adapter(_Target(module)), graph_factory=factory
    )
    start = threading.Barrier(3)
    results: list[tuple[int, torch.Tensor]] = []
    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            torch.cuda.set_device(0)
            start.wait()
            value = torch.full((2,), float(index), device="cuda")
            results.append((index, executor(value)))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()
    torch.cuda.synchronize()

    assert not errors
    assert len(factory.runners) == 2
    for index, output in results:
        assert torch.equal(output, torch.full((2,), float(index * 2), device="cuda"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA devices are unavailable",
)
def test_capture_uses_input_device_when_current_device_differs() -> None:
    previous_device = torch.cuda.current_device()
    target_device = torch.device("cuda:1")
    try:
        torch.cuda.set_device(0)
        module = _Scale().to(target_device).eval()
        executor = OpenPIVisualGraphExecutor(
            _Adapter(_Target(module)),
            warmup_iterations=1,
        )
        value = torch.arange(8, device=target_device, dtype=torch.float32)

        first = executor(value)
        second = executor(value + 1)
        torch.cuda.synchronize(target_device)

        assert torch.equal(first, value * 2)
        assert torch.equal(second, (value + 1) * 2)
        assert executor.diagnostics().counters["captures_succeeded"] == 1
    finally:
        torch.cuda.set_device(previous_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_graph_replays_only_after_bit_exact_first_call() -> None:
    module = _Scale().cuda().eval()
    adapter = _Adapter(_Target(module))
    executor = OpenPIVisualGraphExecutor(
        adapter,
        warmup_iterations=1,
        cache_capacity=1,
    )
    first_input = torch.arange(8, device="cuda", dtype=torch.float32)
    second_input = first_input + 1

    first = executor(first_input)
    second = executor(second_input)
    torch.cuda.synchronize()

    assert torch.equal(first, first_input * 2)
    assert torch.equal(second, second_input * 2)
    diagnostics = executor.diagnostics()
    assert diagnostics.counters["captures_succeeded"] == 1
    assert diagnostics.counters["graph_replays"] == 1
    assert diagnostics.last_decision.path == "cuda_graph"


class _PinnedVision(torch.nn.Module):
    def forward(self, image: torch.Tensor):
        return type("VisionOutput", (), {"last_hidden_state": image + 2})()


class _PinnedProjector(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 3


class _PinnedPaliGemmaOwner(torch.nn.Module):
    __module__ = "openpi.models_pytorch.gemma_pytorch"

    def __init__(self) -> None:
        super().__init__()
        model = torch.nn.Module()
        model.vision_tower = _PinnedVision()
        model.multi_modal_projector = _PinnedProjector()
        paligemma = torch.nn.Module()
        paligemma.model = model
        self.paligemma = paligemma

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.paligemma.model.vision_tower(image).last_hidden_state
        return self.paligemma.model.multi_modal_projector(hidden)


def test_concrete_rlinf_adapter_patches_real_embed_image_and_restores() -> None:
    owner = _PinnedPaliGemmaOwner().eval()
    model = type("OpenPIModel", (), {"paligemma_with_expert": owner})()
    adapter = RLinfOpenPIVisualGraphAdapter(model)
    factory = _FakeFactory()
    executor = OpenPIVisualGraphExecutor(adapter, graph_factory=factory)
    patch = InstalledRLinfOpenPIVisualGraph(
        adapter,
        executor,
        default_context=OpenPIVisualExecutionContext(dynamic_branch=True),
    ).install()
    image = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

    output = owner.embed_image(image)
    torch.testing.assert_close(output, (image + 2) * 3)
    assert executor.diagnostics().last_decision.reason == "dynamic_branch_request"

    with patch.execution_context(OpenPIVisualExecutionContext(rtc=True)):
        torch.testing.assert_close(owner.embed_image(image + 1), (image + 3) * 3)
        assert executor.diagnostics().last_decision.reason == "rtc_request"

    patch.uninstall()
    assert callable(owner.embed_image)
    torch.testing.assert_close(owner.embed_image(image), (image + 2) * 3)


def test_concrete_rlinf_adapter_rejects_unpinned_owner_layout() -> None:
    owner = _PinnedPaliGemmaOwner()
    type(owner).__module__ = "untrusted.openpi"
    model = type("OpenPIModel", (), {"paligemma_with_expert": owner})()
    try:
        with pytest.raises(TypeError, match="unsupported OpenPI visual owner"):
            RLinfOpenPIVisualGraphAdapter(model)
    finally:
        type(owner).__module__ = "openpi.models_pytorch.gemma_pytorch"
