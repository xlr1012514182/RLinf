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

"""Optional TensorRT execution and evidence-gated numerical diagnostics.

TensorRT and PyTorch are imported only when a corresponding object requires
them.  The comparison path itself accepts NumPy-compatible CPU arrays.  A
numerical mismatch is an observation, not proof that BF16 fusion caused it;
causal attribution must be supplied with explicit evidence markers.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..errors import OptionalDependencyError


@dataclass(frozen=True)
class InferenceTrace:
    """Final output plus optional named intermediate tensors."""

    output: Any
    intermediates: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TraceRunner(Protocol):
    """Common protocol for PyTorch and TensorRT trace runners."""

    def __call__(
        self,
        inputs: Mapping[str, Any],
        *,
        capture_intermediates: bool = False,
    ) -> InferenceTrace:
        """Execute one inference and return a trace."""


@runtime_checkable
class TensorRTExecutor(Protocol):
    """Adapter responsible for bindings, allocation, and optional taps."""

    def __call__(
        self,
        engine: Any,
        inputs: Mapping[str, Any],
        *,
        capture_intermediates: bool = False,
    ) -> InferenceTrace:
        """Execute a deserialized engine."""


class TensorRTEngineRunner:
    """Thin, dependency-lazy wrapper around a TensorRT engine.

    TensorRT layer outputs are not observable unless the built engine exposes
    them.  ``executor`` owns that engine-specific binding policy and must
    return any available taps in :class:`InferenceTrace.intermediates`.
    """

    def __init__(
        self,
        engine: Any,
        executor: TensorRTExecutor,
        *,
        runtime: Any | None = None,
        logger: Any | None = None,
        engine_source: str | None = None,
    ) -> None:
        if engine is None:
            raise ValueError("TensorRT engine must not be None")
        if not callable(executor):
            raise TypeError("TensorRT executor must be callable")
        self.engine = engine
        self.executor = executor
        self.runtime = runtime
        self.logger = logger
        self.engine_source = engine_source

    @classmethod
    def from_serialized_engine(
        cls,
        engine_path: str | Path,
        executor_factory: Callable[[Any], TensorRTExecutor],
        *,
        logger_severity: Any | None = None,
    ) -> TensorRTEngineRunner:
        """Deserialize an engine, importing TensorRT only on this path."""

        path = Path(engine_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            trt = importlib.import_module("tensorrt")
        except (ImportError, OSError) as error:
            raise OptionalDependencyError(
                "deserializing a TensorRT engine requires the tensorrt package"
            ) from error
        severity = trt.Logger.WARNING if logger_severity is None else logger_severity
        logger = trt.Logger(severity)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine {path}")
        executor = executor_factory(engine)
        return cls(
            engine,
            executor,
            runtime=runtime,
            logger=logger,
            engine_source=str(path.resolve()),
        )

    def __call__(
        self,
        inputs: Mapping[str, Any],
        *,
        capture_intermediates: bool = False,
    ) -> InferenceTrace:
        trace = self.executor(
            self.engine,
            inputs,
            capture_intermediates=capture_intermediates,
        )
        if not isinstance(trace, InferenceTrace):
            raise TypeError("TensorRT executor must return InferenceTrace")
        return trace


class ComparisonStatus(str, Enum):
    """Structural status of one tensor comparison."""

    COMPARED = "compared"
    SHAPE_MISMATCH = "shape_mismatch"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class TensorComparison:
    """Strict exactness plus magnitude and ULP-style diagnostics."""

    name: str
    status: ComparisonStatus
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    reference_dtype: str
    candidate_dtype: str
    element_count: int
    dtype_match: bool
    bit_exact: bool
    differing_element_count: int
    nonfinite_mismatch_count: int
    maximum_absolute_error: float
    maximum_relative_error: float
    maximum_ulp_distance: int | None
    ulp_distance_kind: str | None
    note: str = ""


@dataclass(frozen=True)
class _ArrayView:
    numeric: Any
    raw: bytes
    raw_elements: Any
    shape: tuple[int, ...]
    dtype_name: str
    item_bytes: int
    floating: bool
    ulp_bits: Any | None
    ulp_width: int | None


def _numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except (ImportError, OSError) as error:
        raise OptionalDependencyError(
            "TensorRT exactness diagnostics require the numpy package"
        ) from error


def _is_torch_tensor(value: Any) -> bool:
    module_name = type(value).__module__
    return module_name == "torch" or module_name.startswith("torch.")


def _as_array_view(value: Any) -> _ArrayView:
    np = _numpy()
    if _is_torch_tensor(value):
        tensor = value.detach().cpu().contiguous()
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if dtype_name == "bfloat16":
            numeric = np.ascontiguousarray(tensor.float().numpy())
            raw_words = np.ascontiguousarray(
                tensor.view(importlib.import_module("torch").uint16).numpy()
            )
            raw_elements = raw_words.view(np.uint8).reshape(raw_words.size, 2)
            return _ArrayView(
                numeric=numeric,
                raw=raw_words.tobytes(),
                raw_elements=raw_elements,
                shape=tuple(tensor.shape),
                dtype_name="bfloat16",
                item_bytes=2,
                floating=True,
                ulp_bits=raw_words.reshape(-1),
                ulp_width=16,
            )
        array = np.ascontiguousarray(tensor.numpy())
    else:
        try:
            array = np.ascontiguousarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"value of type {type(value).__name__} is not tensor-like"
            ) from error

    if array.dtype.hasobject:
        raise TypeError("object arrays are unsupported")
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"non-numeric dtype {array.dtype} is unsupported")
    raw_elements = array.view(np.uint8).reshape(array.size, array.dtype.itemsize)
    ulp_bits = None
    ulp_width = None
    if array.dtype == np.dtype("float16"):
        ulp_bits, ulp_width = array.view(np.uint16).reshape(-1), 16
    elif array.dtype == np.dtype("float32"):
        ulp_bits, ulp_width = array.view(np.uint32).reshape(-1), 32
    elif array.dtype == np.dtype("float64"):
        ulp_bits, ulp_width = array.view(np.uint64).reshape(-1), 64
    return _ArrayView(
        numeric=array,
        raw=array.tobytes(),
        raw_elements=raw_elements,
        shape=tuple(array.shape),
        dtype_name=str(array.dtype),
        item_bytes=array.dtype.itemsize,
        floating=bool(np.issubdtype(array.dtype, np.floating)),
        ulp_bits=ulp_bits,
        ulp_width=ulp_width,
    )


def _ordered_float_bits(bits: Any, width: int) -> Any:
    np = _numpy()
    sign_mask = np.array(1 << (width - 1), dtype=bits.dtype)
    return np.where((bits & sign_mask) != 0, ~bits, bits ^ sign_mask)


def _maximum_ulp_distance(
    reference: _ArrayView,
    candidate: _ArrayView,
    finite_mask: Any,
) -> int | None:
    np = _numpy()
    if (
        reference.ulp_bits is None
        or candidate.ulp_bits is None
        or reference.ulp_width != candidate.ulp_width
        or reference.dtype_name != candidate.dtype_name
    ):
        return None
    mask = np.asarray(finite_mask).reshape(-1)
    if not bool(mask.any()):
        return None
    reference_ordered = _ordered_float_bits(
        reference.ulp_bits[mask], reference.ulp_width
    )
    candidate_ordered = _ordered_float_bits(
        candidate.ulp_bits[mask], candidate.ulp_width
    )
    upper = np.maximum(reference_ordered, candidate_ordered)
    lower = np.minimum(reference_ordered, candidate_ordered)
    return int((upper - lower).max(initial=0))


def _byte_mismatch_by_element(reference: _ArrayView, candidate: _ArrayView) -> Any:
    np = _numpy()
    if reference.item_bytes != candidate.item_bytes:
        return np.ones(math.prod(reference.shape), dtype=bool)
    return np.any(reference.raw_elements != candidate.raw_elements, axis=1)


def compare_tensor(name: str, reference: Any, candidate: Any) -> TensorComparison:
    """Compare tensor-like values without accepting a numerical tolerance."""

    np = _numpy()
    try:
        reference_view = _as_array_view(reference)
        candidate_view = _as_array_view(candidate)
    except TypeError as error:
        return TensorComparison(
            name=name,
            status=ComparisonStatus.UNSUPPORTED,
            reference_shape=(),
            candidate_shape=(),
            reference_dtype=type(reference).__name__,
            candidate_dtype=type(candidate).__name__,
            element_count=0,
            dtype_match=False,
            bit_exact=False,
            differing_element_count=0,
            nonfinite_mismatch_count=0,
            maximum_absolute_error=math.inf,
            maximum_relative_error=math.inf,
            maximum_ulp_distance=None,
            ulp_distance_kind=None,
            note=str(error),
        )
    dtype_match = reference_view.dtype_name == candidate_view.dtype_name
    if reference_view.shape != candidate_view.shape:
        return TensorComparison(
            name=name,
            status=ComparisonStatus.SHAPE_MISMATCH,
            reference_shape=reference_view.shape,
            candidate_shape=candidate_view.shape,
            reference_dtype=reference_view.dtype_name,
            candidate_dtype=candidate_view.dtype_name,
            element_count=math.prod(reference_view.shape),
            dtype_match=dtype_match,
            bit_exact=False,
            differing_element_count=math.prod(reference_view.shape),
            nonfinite_mismatch_count=0,
            maximum_absolute_error=math.inf,
            maximum_relative_error=math.inf,
            maximum_ulp_distance=None,
            ulp_distance_kind=None,
            note="shape mismatch prevents elementwise diagnostics",
        )

    element_count = math.prod(reference_view.shape)
    byte_mismatch = _byte_mismatch_by_element(reference_view, candidate_view)
    bit_exact = dtype_match and reference_view.raw == candidate_view.raw
    reference_numeric = np.asarray(reference_view.numeric, dtype=np.float64)
    candidate_numeric = np.asarray(candidate_view.numeric, dtype=np.float64)
    reference_finite = np.isfinite(reference_numeric)
    candidate_finite = np.isfinite(candidate_numeric)
    both_finite = reference_finite & candidate_finite
    nonfinite_equal = (
        (~reference_finite)
        & (~candidate_finite)
        & (~byte_mismatch.reshape(reference_view.shape))
    )
    nonfinite_mismatch = (~both_finite) & (~nonfinite_equal)
    nonfinite_mismatch_count = int(np.count_nonzero(nonfinite_mismatch))

    if bool(both_finite.any()):
        difference = np.abs(
            reference_numeric[both_finite] - candidate_numeric[both_finite]
        )
        maximum_absolute_error = float(difference.max(initial=0.0))
        denominator = np.maximum(
            np.abs(reference_numeric[both_finite]), np.finfo(np.float64).tiny
        )
        maximum_relative_error = float((difference / denominator).max(initial=0.0))
    else:
        maximum_absolute_error = 0.0
        maximum_relative_error = 0.0
    if nonfinite_mismatch_count:
        maximum_absolute_error = math.inf
        maximum_relative_error = math.inf

    maximum_ulp_distance = _maximum_ulp_distance(
        reference_view, candidate_view, both_finite
    )
    ulp_kind = (
        f"ordered-{reference_view.dtype_name}-bit-distance"
        if maximum_ulp_distance is not None
        else None
    )
    return TensorComparison(
        name=name,
        status=ComparisonStatus.COMPARED,
        reference_shape=reference_view.shape,
        candidate_shape=candidate_view.shape,
        reference_dtype=reference_view.dtype_name,
        candidate_dtype=candidate_view.dtype_name,
        element_count=element_count,
        dtype_match=dtype_match,
        bit_exact=bit_exact,
        differing_element_count=int(np.count_nonzero(byte_mismatch)),
        nonfinite_mismatch_count=nonfinite_mismatch_count,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        maximum_ulp_distance=maximum_ulp_distance,
        ulp_distance_kind=ulp_kind,
    )


class AttributionStatus(str, Enum):
    """Evidence state for a proposed BF16-fusion causal attribution."""

    NOT_EVALUATED = "not_evaluated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SUPPORTED = "supported"
    REFUTED = "refuted"


@dataclass(frozen=True)
class EvidenceMarker:
    """Exact locator and observation supporting an attribution assessment."""

    evidence_id: str
    kind: str
    locator: str
    observation: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.evidence_id, self.kind, self.locator, self.observation)
        ):
            raise ValueError("all evidence marker fields must be non-empty")


@dataclass(frozen=True)
class BF16FusionAttribution:
    """Evidence-gated assessment; never inferred from output drift alone."""

    status: AttributionStatus = AttributionStatus.NOT_EVALUATED
    rationale: str = (
        "Numerical drift alone is insufficient to attribute an error to BF16 fusion."
    )
    evidence: tuple[EvidenceMarker, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, AttributionStatus):
            object.__setattr__(self, "status", AttributionStatus(self.status))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.rationale.strip():
            raise ValueError("attribution rationale must not be empty")
        if not all(isinstance(marker, EvidenceMarker) for marker in self.evidence):
            raise TypeError("attribution evidence must contain EvidenceMarker objects")
        if (
            self.status in {AttributionStatus.SUPPORTED, AttributionStatus.REFUTED}
            and not self.evidence
        ):
            raise ValueError(
                "supported or refuted BF16-fusion attribution requires evidence markers"
            )


@dataclass(frozen=True)
class InferenceAuditReport:
    """PyTorch-vs-TensorRT structural and numerical comparison."""

    reference_backend: str
    candidate_backend: str
    final_output: TensorComparison
    intermediate_layers: tuple[TensorComparison, ...]
    missing_candidate_layers: tuple[str, ...]
    extra_candidate_layers: tuple[str, ...]
    intermediates_bit_exact: bool
    overall_bit_exact: bool
    bf16_fusion_attribution: BF16FusionAttribution
    reference_metadata: Mapping[str, Any]
    candidate_metadata: Mapping[str, Any]


def compare_inference_traces(
    reference: InferenceTrace,
    candidate: InferenceTrace,
    *,
    reference_backend: str = "pytorch",
    candidate_backend: str = "tensorrt",
    bf16_fusion_attribution: BF16FusionAttribution | None = None,
) -> InferenceAuditReport:
    """Compare named intermediate layers and the final output."""

    reference_keys = set(reference.intermediates)
    candidate_keys = set(candidate.intermediates)
    shared_keys = sorted(reference_keys & candidate_keys)
    intermediate_layers = tuple(
        compare_tensor(
            layer_name,
            reference.intermediates[layer_name],
            candidate.intermediates[layer_name],
        )
        for layer_name in shared_keys
    )
    missing_candidate_layers = tuple(sorted(reference_keys - candidate_keys))
    extra_candidate_layers = tuple(sorted(candidate_keys - reference_keys))
    intermediates_bit_exact = (
        not missing_candidate_layers
        and not extra_candidate_layers
        and all(layer.bit_exact for layer in intermediate_layers)
    )
    final_output = compare_tensor("output", reference.output, candidate.output)
    attribution = bf16_fusion_attribution or BF16FusionAttribution()
    return InferenceAuditReport(
        reference_backend=reference_backend,
        candidate_backend=candidate_backend,
        final_output=final_output,
        intermediate_layers=intermediate_layers,
        missing_candidate_layers=missing_candidate_layers,
        extra_candidate_layers=extra_candidate_layers,
        intermediates_bit_exact=intermediates_bit_exact,
        overall_bit_exact=intermediates_bit_exact and final_output.bit_exact,
        bf16_fusion_attribution=attribution,
        reference_metadata=reference.metadata,
        candidate_metadata=candidate.metadata,
    )


def audit_pytorch_vs_tensorrt(
    pytorch_runner: TraceRunner,
    tensorrt_runner: TraceRunner,
    inputs: Mapping[str, Any],
    *,
    capture_intermediates: bool = True,
    bf16_fusion_attribution: BF16FusionAttribution | None = None,
) -> InferenceAuditReport:
    """Execute paired runners once and compare their returned traces."""

    reference = pytorch_runner(inputs, capture_intermediates=capture_intermediates)
    candidate = tensorrt_runner(inputs, capture_intermediates=capture_intermediates)
    if not isinstance(reference, InferenceTrace):
        raise TypeError("PyTorch runner must return InferenceTrace")
    if not isinstance(candidate, InferenceTrace):
        raise TypeError("TensorRT runner must return InferenceTrace")
    return compare_inference_traces(
        reference,
        candidate,
        bf16_fusion_attribution=bf16_fusion_attribution,
    )


__all__ = [
    "AttributionStatus",
    "BF16FusionAttribution",
    "ComparisonStatus",
    "EvidenceMarker",
    "InferenceAuditReport",
    "InferenceTrace",
    "TensorComparison",
    "TensorRTEngineRunner",
    "TensorRTExecutor",
    "TraceRunner",
    "audit_pytorch_vs_tensorrt",
    "compare_inference_traces",
    "compare_tensor",
]
