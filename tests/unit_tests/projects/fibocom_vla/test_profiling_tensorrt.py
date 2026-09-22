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

"""CPU-only tests for profiling and TensorRT evidence diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from rlinf.projects.fibocom_vla.errors import OptionalDependencyError
from rlinf.projects.fibocom_vla.inference import profiling
from rlinf.projects.fibocom_vla.inference.profiling import (
    ExternalTimingEvidence,
    InferencePath,
    StageProfiler,
    StageSpec,
    paired_alternating_benchmark,
    profile_stages,
)
from rlinf.projects.fibocom_vla.inference.tensorrt_audit import (
    AttributionStatus,
    BF16FusionAttribution,
    EvidenceMarker,
    InferenceTrace,
    TensorRTEngineRunner,
    audit_pytorch_vs_tensorrt,
    compare_inference_traces,
    compare_tensor,
)


class _IncrementingClock:
    def __init__(self, step_ns: int) -> None:
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        value = self.value
        self.value += self.step_ns
        return value


class _AdvanceableClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        return self.value

    def advance(self, duration_ns: int) -> None:
        self.value += duration_ns


class _FakeCudaEvent:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda
        self.timestamp_ms = 0.0

    def record(self) -> None:
        self.timestamp_ms = self.cuda.timestamp_ms

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, other: _FakeCudaEvent) -> float:
        return other.timestamp_ms - self.timestamp_ms


class _FakeCuda:
    def __init__(self) -> None:
        self.timestamp_ms = 0.0

    def Event(self, *, enable_timing: bool) -> _FakeCudaEvent:  # noqa: N802
        assert enable_timing is True
        return _FakeCudaEvent(self)

    def synchronize(self) -> None:
        return None


def test_profile_stages_warms_up_and_labels_cpu_paths() -> None:
    calls = {"visual": 0, "language_gemm": 0}

    def visual() -> None:
        calls["visual"] += 1

    def language_gemm() -> None:
        calls["language_gemm"] += 1

    report = profile_stages(
        [
            StageSpec("visual", visual, (InferencePath.VISUAL,)),
            StageSpec(
                "language_gemm",
                language_gemm,
                (InferencePath.LANGUAGE, InferencePath.GEMM),
            ),
        ],
        warmup_iterations=2,
        measured_iterations=3,
        use_cuda=False,
        clock_ns=_IncrementingClock(1_000_000),
    )

    assert calls == {"visual": 5, "language_gemm": 5}
    assert report.cuda_enabled is False
    assert len(report.samples) == 6
    assert report.stage("visual").paths == (InferencePath.VISUAL,)
    assert report.stage("language_gemm").paths == (
        InferencePath.LANGUAGE,
        InferencePath.GEMM,
    )
    assert report.stage("visual").cpu.p50_ms == pytest.approx(1.0)
    assert report.stage("visual").cpu.p95_ms == pytest.approx(1.0)
    assert report.stage("visual").cuda is None


def test_stage_profiler_records_cuda_events_without_requiring_a_gpu(
    monkeypatch,
) -> None:
    fake_cuda = _FakeCuda()

    def gemm() -> None:
        fake_cuda.timestamp_ms += 1.5

    profiler = StageProfiler(
        warmup_iterations=1,
        measured_iterations=3,
        use_cuda=True,
        clock_ns=_IncrementingClock(500_000),
    )
    monkeypatch.setattr(profiler, "_resolve_cuda", lambda: fake_cuda)

    report = profiler.profile([StageSpec("gemm", gemm, (InferencePath.GEMM,))])

    assert report.cuda_enabled is True
    assert report.stage("gemm").cpu.p50_ms == pytest.approx(0.5)
    assert report.stage("gemm").cuda is not None
    assert report.stage("gemm").cuda.p50_ms == pytest.approx(1.5)
    assert report.stage("gemm").cuda.p95_ms == pytest.approx(1.5)


def test_paired_benchmark_alternates_ab_ba_and_preserves_negative_delta() -> None:
    clock = _AdvanceableClock()

    def baseline() -> None:
        clock.advance(4_000_000)

    def candidate() -> None:
        clock.advance(2_000_000)

    report = paired_alternating_benchmark(
        baseline,
        candidate,
        benchmark_name="pi05_runtime_candidate",
        paths=(InferencePath.LANGUAGE, InferencePath.GEMM),
        warmup_pairs=1,
        measured_pairs=4,
        clock_ns=clock,
    )

    assert tuple(sample.order for sample in report.samples) == ("AB", "BA", "AB", "BA")
    assert report.baseline.p50_ms == pytest.approx(4.0)
    assert report.baseline.p95_ms == pytest.approx(4.0)
    assert report.candidate.p50_ms == pytest.approx(2.0)
    assert report.candidate_minus_baseline.p50_ms == pytest.approx(-2.0)
    assert report.median_speedup == pytest.approx(2.0)


def test_external_timing_annotation_does_not_supply_benchmark_samples() -> None:
    assert all(hasattr(profiling, name) for name in profiling.__all__)
    evidence = ExternalTimingEvidence(
        evidence_id="synthetic-unit-test-only",
        system="synthetic external fixture; not a model measurement",
        optimization="synthetic candidate",
        baseline_ms=999.0,
        candidate_ms=111.0,
        timing_boundary="synthetic test fixture",
        exactness_observation="not evaluated",
        source_locator="test_external_timing_annotation_does_not_supply_benchmark_samples",
    )

    assert evidence.eligible_as_default is False
    report = paired_alternating_benchmark(
        lambda: None,
        lambda: None,
        benchmark_name="synthetic-current-invocation",
        paths=(InferencePath.VISUAL,),
        warmup_pairs=0,
        measured_pairs=2,
        clock_ns=_IncrementingClock(1_000_000),
    )

    assert report.baseline.p50_ms == pytest.approx(1.0)
    assert report.candidate.p50_ms == pytest.approx(1.0)
    assert report.baseline.p50_ms != evidence.baseline_ms
    assert report.candidate.p50_ms != evidence.candidate_ms


def test_tensor_comparison_reports_exact_abs_rel_and_ulp_diagnostics() -> None:
    reference = np.array([1.0, -0.0], dtype=np.float32)
    candidate = np.array(
        [np.nextafter(np.float32(1.0), np.float32(2.0)), 0.0],
        dtype=np.float32,
    )

    comparison = compare_tensor("layer", reference, candidate)

    assert comparison.bit_exact is False
    assert comparison.differing_element_count == 2
    assert comparison.maximum_absolute_error == pytest.approx(np.finfo(np.float32).eps)
    assert comparison.maximum_relative_error == pytest.approx(np.finfo(np.float32).eps)
    assert comparison.maximum_ulp_distance == 1
    assert comparison.ulp_distance_kind == "ordered-float32-bit-distance"


def test_tensor_comparison_preserves_bfloat16_bits_on_cpu() -> None:
    reference = torch.tensor([1.0], dtype=torch.bfloat16)
    candidate = torch.tensor([1.0078125], dtype=torch.bfloat16)

    comparison = compare_tensor("bf16_layer", reference, candidate)

    assert comparison.reference_dtype == "bfloat16"
    assert comparison.bit_exact is False
    assert comparison.differing_element_count == 1
    assert comparison.maximum_absolute_error == pytest.approx(0.0078125)
    assert comparison.maximum_ulp_distance == 1
    assert comparison.ulp_distance_kind == "ordered-bfloat16-bit-distance"


def test_trace_audit_compares_layers_and_does_not_infer_bf16_fusion() -> None:
    reference = InferenceTrace(
        output=np.array([1.0], dtype=np.float32),
        intermediates={
            "visual": np.array([2.0], dtype=np.float32),
            "language.mlp": np.array([1.0], dtype=np.float32),
        },
    )
    candidate = InferenceTrace(
        output=np.array([np.nextafter(np.float32(1.0), np.float32(2.0))]),
        intermediates={
            "visual": np.array([2.0], dtype=np.float32),
            "language.mlp": np.array([np.nextafter(np.float32(1.0), np.float32(2.0))]),
        },
    )

    report = compare_inference_traces(reference, candidate)

    assert report.overall_bit_exact is False
    assert report.final_output.maximum_ulp_distance == 1
    assert report.intermediate_layers[0].name == "language.mlp"
    assert report.intermediate_layers[0].bit_exact is False
    assert report.intermediate_layers[1].name == "visual"
    assert report.intermediate_layers[1].bit_exact is True
    assert report.bf16_fusion_attribution.status is AttributionStatus.NOT_EVALUATED
    assert "insufficient" in report.bf16_fusion_attribution.rationale


def test_bf16_fusion_conclusion_requires_evidence_marker() -> None:
    with pytest.raises(ValueError, match="requires evidence markers"):
        BF16FusionAttribution(
            status=AttributionStatus.SUPPORTED,
            rationale="fusion caused the observed layer divergence",
        )

    marker = EvidenceMarker(
        evidence_id="ablation-001",
        kind="paired_layer_ablation",
        locator="artifacts/trt_ablation.json:layer_17",
        observation="disabling one fusion restores the layer output",
    )
    attribution = BF16FusionAttribution(
        status=AttributionStatus.SUPPORTED,
        rationale="paired engine ablation isolates the candidate fusion",
        evidence=(marker,),
    )
    assert attribution.evidence == (marker,)

    with pytest.raises(ValueError, match="requires evidence markers"):
        BF16FusionAttribution(
            status="supported",  # type: ignore[arg-type]
            rationale="a string status must not bypass the evidence gate",
        )


def test_injected_tensorrt_runner_never_imports_tensorrt(monkeypatch) -> None:
    from rlinf.projects.fibocom_vla.inference import tensorrt_audit

    original_import = tensorrt_audit.importlib.import_module

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tensorrt":
            raise AssertionError("direct engine injection must not import TensorRT")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(tensorrt_audit.importlib, "import_module", guarded_import)

    def executor(
        engine: Any,
        inputs: dict[str, Any],
        *,
        capture_intermediates: bool = False,
    ) -> InferenceTrace:
        assert engine == "fake-engine"
        return InferenceTrace(
            output=inputs["x"],
            intermediates={"tap": inputs["x"]} if capture_intermediates else {},
        )

    runner = TensorRTEngineRunner("fake-engine", executor)
    trace = runner(
        {"x": np.array([3.0], dtype=np.float32)},
        capture_intermediates=True,
    )
    assert trace.intermediates.keys() == {"tap"}


def test_deserialization_reports_missing_optional_tensorrt(
    monkeypatch, tmp_path
) -> None:
    from rlinf.projects.fibocom_vla.inference import tensorrt_audit

    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine-placeholder")
    original_import = tensorrt_audit.importlib.import_module

    def missing_tensorrt(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tensorrt":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(tensorrt_audit.importlib, "import_module", missing_tensorrt)
    with pytest.raises(OptionalDependencyError, match="tensorrt package"):
        TensorRTEngineRunner.from_serialized_engine(engine_path, lambda engine: engine)


def test_paired_runner_audit_passes_capture_request_and_reports_exactness() -> None:
    capture_requests: list[bool] = []

    def runner(
        inputs: dict[str, Any],
        *,
        capture_intermediates: bool = False,
    ) -> InferenceTrace:
        capture_requests.append(capture_intermediates)
        return InferenceTrace(
            output=inputs["x"],
            intermediates={"layer": inputs["x"]} if capture_intermediates else {},
        )

    report = audit_pytorch_vs_tensorrt(
        runner,
        runner,
        {"x": np.array([1.0], dtype=np.float32)},
    )

    assert capture_requests == [True, True]
    assert report.overall_bit_exact is True
    assert report.final_output.maximum_absolute_error == 0.0


def test_shape_mismatch_is_not_bit_exact() -> None:
    comparison = compare_tensor(
        "output",
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
    )

    assert comparison.bit_exact is False
    assert comparison.maximum_absolute_error == pytest.approx(float("inf"))
