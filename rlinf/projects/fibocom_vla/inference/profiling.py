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

"""Measured latency profiling for VLA inference paths.

Benchmark reports contain samples collected by the current invocation.
External timing annotations can be supplied explicitly with
``ExternalTimingEvidence``; no historical measurements are bundled here.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..errors import OptionalDependencyError


class InferencePath(str, Enum):
    """Controlled labels for regions in a VLA inference pipeline."""

    VISUAL = "visual"
    LANGUAGE = "language"
    GEMM = "gemm"
    ACTION = "action"
    RUNTIME = "runtime"
    OTHER = "other"


# ``PathLabel`` is a concise public synonym used by benchmark clients.
PathLabel = InferencePath


@dataclass(frozen=True)
class ExternalTimingEvidence:
    """A non-portable timing observation from an explicitly named system."""

    evidence_id: str
    system: str
    optimization: str
    baseline_ms: float
    candidate_ms: float
    timing_boundary: str
    exactness_observation: str
    source_locator: str
    eligible_as_default: bool = False


def _coerce_path(value: InferencePath | str) -> InferencePath:
    if isinstance(value, InferencePath):
        return value
    try:
        return InferencePath(value)
    except ValueError as error:
        allowed = ", ".join(path.value for path in InferencePath)
        raise ValueError(
            f"unknown inference path {value!r}; expected one of {allowed}"
        ) from error


def _normalise_paths(
    values: Iterable[InferencePath | str],
) -> tuple[InferencePath, ...]:
    paths = tuple(dict.fromkeys(_coerce_path(value) for value in values))
    if not paths:
        raise ValueError("at least one inference path label is required")
    return paths


@dataclass(frozen=True)
class StageSpec:
    """One callable stage and the controlled inference paths it represents."""

    name: str
    function: Callable[[], Any]
    paths: tuple[InferencePath | str, ...] = (InferencePath.OTHER,)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage name must not be empty")
        if not callable(self.function):
            raise TypeError("stage function must be callable")
        object.__setattr__(self, "paths", _normalise_paths(self.paths))


@dataclass(frozen=True)
class LatencySummary:
    """Distribution summary in milliseconds."""

    count: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_latencies(
    samples_ms: Iterable[float],
    *,
    allow_negative: bool = False,
) -> LatencySummary:
    """Return p50/p95 and range statistics for observed samples."""

    samples = tuple(float(sample) for sample in samples_ms)
    if not samples:
        raise ValueError("at least one latency sample is required")
    if any(not math.isfinite(sample) for sample in samples):
        raise ValueError("latency samples must be finite")
    if not allow_negative and any(sample < 0.0 for sample in samples):
        raise ValueError("latency samples must be non-negative")
    ordered = sorted(samples)
    return LatencySummary(
        count=len(samples),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        mean_ms=sum(samples) / len(samples),
        minimum_ms=ordered[0],
        maximum_ms=ordered[-1],
    )


# Keep the British spelling as an explicit compatibility alias.
summarise_latencies = summarize_latencies


@dataclass(frozen=True)
class StageSample:
    """One measured stage invocation."""

    iteration: int
    stage_name: str
    paths: tuple[InferencePath, ...]
    cpu_ms: float
    cuda_ms: float | None


@dataclass(frozen=True)
class StageSummary:
    """Aggregated CPU and optional CUDA timing for one stage."""

    stage_name: str
    paths: tuple[InferencePath, ...]
    cpu: LatencySummary
    cuda: LatencySummary | None


@dataclass(frozen=True)
class StageProfileReport:
    """Measured samples and summaries for a phased inference path."""

    warmup_iterations: int
    measured_iterations: int
    cuda_enabled: bool
    samples: tuple[StageSample, ...]
    stages: tuple[StageSummary, ...]

    def stage(self, name: str) -> StageSummary:
        """Return a named stage or raise a descriptive lookup error."""

        for stage in self.stages:
            if stage.stage_name == name:
                return stage
        raise KeyError(f"unknown stage {name!r}")


class StageProfiler:
    """Warm up and profile CPU dispatch plus optional CUDA event latency."""

    def __init__(
        self,
        *,
        warmup_iterations: int = 5,
        measured_iterations: int = 20,
        use_cuda: bool | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if measured_iterations < 1:
            raise ValueError("measured_iterations must be positive")
        self.warmup_iterations = warmup_iterations
        self.measured_iterations = measured_iterations
        self.use_cuda = use_cuda
        self.clock_ns = clock_ns

    def _resolve_cuda(self) -> Any | None:
        if self.use_cuda is False:
            return None
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError) as error:
            if self.use_cuda:
                raise OptionalDependencyError(
                    "CUDA profiling requires the optional PyTorch dependency"
                ) from error
            return None
        cuda = torch.cuda
        if not cuda.is_available():
            if self.use_cuda:
                raise OptionalDependencyError(
                    "CUDA profiling was requested but torch.cuda is unavailable"
                )
            return None
        return cuda

    def profile(self, stages: Sequence[StageSpec]) -> StageProfileReport:
        """Execute ordered stages with warmup and per-stage measurements."""

        stage_specs = tuple(stages)
        if not stage_specs:
            raise ValueError("at least one stage is required")
        names = tuple(stage.name for stage in stage_specs)
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        cuda = self._resolve_cuda()

        for _ in range(self.warmup_iterations):
            for stage in stage_specs:
                stage.function()
        if cuda is not None:
            cuda.synchronize()

        samples: list[StageSample] = []
        for iteration in range(self.measured_iterations):
            for stage in stage_specs:
                start_event = end_event = None
                if cuda is not None:
                    start_event = cuda.Event(enable_timing=True)
                    end_event = cuda.Event(enable_timing=True)
                    start_event.record()
                cpu_start = self.clock_ns()
                stage.function()
                cpu_end = self.clock_ns()
                cuda_ms = None
                if start_event is not None and end_event is not None:
                    end_event.record()
                    end_event.synchronize()
                    cuda_ms = float(start_event.elapsed_time(end_event))
                samples.append(
                    StageSample(
                        iteration=iteration,
                        stage_name=stage.name,
                        paths=stage.paths,
                        cpu_ms=(cpu_end - cpu_start) / 1_000_000.0,
                        cuda_ms=cuda_ms,
                    )
                )

        summaries: list[StageSummary] = []
        for stage in stage_specs:
            stage_samples = tuple(
                sample for sample in samples if sample.stage_name == stage.name
            )
            cuda_samples = tuple(
                sample.cuda_ms for sample in stage_samples if sample.cuda_ms is not None
            )
            summaries.append(
                StageSummary(
                    stage_name=stage.name,
                    paths=stage.paths,
                    cpu=summarize_latencies(sample.cpu_ms for sample in stage_samples),
                    cuda=(summarize_latencies(cuda_samples) if cuda_samples else None),
                )
            )
        return StageProfileReport(
            warmup_iterations=self.warmup_iterations,
            measured_iterations=self.measured_iterations,
            cuda_enabled=cuda is not None,
            samples=tuple(samples),
            stages=tuple(summaries),
        )


def profile_stages(
    stages: Sequence[StageSpec],
    *,
    warmup_iterations: int = 5,
    measured_iterations: int = 20,
    use_cuda: bool | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> StageProfileReport:
    """Convenience wrapper around :class:`StageProfiler`."""

    return StageProfiler(
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        use_cuda=use_cuda,
        clock_ns=clock_ns,
    ).profile(stages)


@dataclass(frozen=True)
class PairedSample:
    """One alternating baseline/candidate latency pair."""

    pair_index: int
    order: str
    baseline_ms: float
    candidate_ms: float

    @property
    def candidate_minus_baseline_ms(self) -> float:
        return self.candidate_ms - self.baseline_ms


@dataclass(frozen=True)
class PairedBenchmarkReport:
    """Alternating paired benchmark with no synthetic timing samples."""

    benchmark_name: str
    paths: tuple[InferencePath, ...]
    warmup_pairs: int
    measured_pairs: int
    samples: tuple[PairedSample, ...]
    baseline: LatencySummary
    candidate: LatencySummary
    candidate_minus_baseline: LatencySummary
    median_speedup: float


def _invoke_timed(
    function: Callable[[], Any],
    *,
    clock_ns: Callable[[], int],
    synchronize: Callable[[], Any] | None,
) -> float:
    if synchronize is not None:
        synchronize()
    start = clock_ns()
    function()
    if synchronize is not None:
        synchronize()
    end = clock_ns()
    duration_ms = (end - start) / 1_000_000.0
    if duration_ms < 0.0:
        raise RuntimeError("benchmark clock moved backwards")
    return duration_ms


def paired_alternating_benchmark(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    benchmark_name: str = "vla_inference",
    paths: Iterable[InferencePath | str] = (InferencePath.RUNTIME,),
    warmup_pairs: int = 5,
    measured_pairs: int = 20,
    synchronize: Callable[[], Any] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> PairedBenchmarkReport:
    """Measure AB/BA pairs to reduce order and thermal drift confounding.

    ``synchronize`` may be ``torch.cuda.synchronize`` for end-to-end GPU
    latency.  CUDA remains an optional dependency because the callback is
    supplied by the caller.
    """

    if not callable(baseline) or not callable(candidate):
        raise TypeError("baseline and candidate must be callable")
    if not benchmark_name.strip():
        raise ValueError("benchmark_name must not be empty")
    if warmup_pairs < 0:
        raise ValueError("warmup_pairs must be non-negative")
    if measured_pairs < 1:
        raise ValueError("measured_pairs must be positive")
    normalised_paths = _normalise_paths(paths)

    for pair_index in range(warmup_pairs):
        functions = (
            (baseline, candidate) if pair_index % 2 == 0 else (candidate, baseline)
        )
        for function in functions:
            function()
            if synchronize is not None:
                synchronize()

    samples: list[PairedSample] = []
    for pair_index in range(measured_pairs):
        if pair_index % 2 == 0:
            baseline_ms = _invoke_timed(
                baseline, clock_ns=clock_ns, synchronize=synchronize
            )
            candidate_ms = _invoke_timed(
                candidate, clock_ns=clock_ns, synchronize=synchronize
            )
            order = "AB"
        else:
            candidate_ms = _invoke_timed(
                candidate, clock_ns=clock_ns, synchronize=synchronize
            )
            baseline_ms = _invoke_timed(
                baseline, clock_ns=clock_ns, synchronize=synchronize
            )
            order = "BA"
        samples.append(
            PairedSample(
                pair_index=pair_index,
                order=order,
                baseline_ms=baseline_ms,
                candidate_ms=candidate_ms,
            )
        )

    baseline_summary = summarize_latencies(sample.baseline_ms for sample in samples)
    candidate_summary = summarize_latencies(sample.candidate_ms for sample in samples)
    median_speedup = (
        math.inf
        if candidate_summary.p50_ms == 0.0
        else baseline_summary.p50_ms / candidate_summary.p50_ms
    )
    return PairedBenchmarkReport(
        benchmark_name=benchmark_name,
        paths=normalised_paths,
        warmup_pairs=warmup_pairs,
        measured_pairs=measured_pairs,
        samples=tuple(samples),
        baseline=baseline_summary,
        candidate=candidate_summary,
        candidate_minus_baseline=summarize_latencies(
            (sample.candidate_minus_baseline_ms for sample in samples),
            allow_negative=True,
        ),
        median_speedup=median_speedup,
    )


__all__ = [
    "ExternalTimingEvidence",
    "InferencePath",
    "LatencySummary",
    "PairedBenchmarkReport",
    "PairedSample",
    "PathLabel",
    "StageProfileReport",
    "StageProfiler",
    "StageSample",
    "StageSpec",
    "StageSummary",
    "paired_alternating_benchmark",
    "profile_stages",
    "summarise_latencies",
    "summarize_latencies",
]
