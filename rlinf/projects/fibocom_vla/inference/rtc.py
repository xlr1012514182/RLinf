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

"""Asynchronous chunk planning and backend-neutral RTC conditioning.

This module owns host scheduling and a deliberately labelled output-space
fallback.  It does *not* claim that output blending is native RTC guidance.
The differentiable hard-prefix/VJP implementation that runs inside the flow
denoising loop lives in :mod:`.rtc_torch` and is reached through
``NativeRTCPolicy.predict_with_rtc``.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..config import RTCConfig
from ..contracts import ActionChunk, ChunkPolicy, Observation, PolicyOutput
from ..errors import ShapeMismatchError
from ..metrics import RuntimeMetrics


@dataclass(frozen=True)
class RTCConditioning:
    """Hard prefix and exponentially decayed overlap target for denoising."""

    hard_prefix: NDArray[np.float32]
    overlap_target: NDArray[np.float32]
    overlap_weights: NDArray[np.float32]
    model_hard_prefix: NDArray[np.float32] | None = None
    model_overlap_target: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        hard = np.asarray(self.hard_prefix, dtype=np.float32)
        target = np.asarray(self.overlap_target, dtype=np.float32)
        weights = np.asarray(self.overlap_weights, dtype=np.float32)
        if hard.ndim != 2 or target.ndim != 2:
            raise ShapeMismatchError("RTC prefix and overlap must be [T, A]")
        if weights.shape != (target.shape[0],):
            raise ShapeMismatchError("RTC overlap weights must be [overlap_horizon]")
        if hard.shape[1] != target.shape[1]:
            raise ShapeMismatchError("RTC action dimensions differ")
        if not (
            np.all(np.isfinite(hard))
            and np.all(np.isfinite(target))
            and np.all(np.isfinite(weights))
        ):
            raise ShapeMismatchError("RTC conditioning contains non-finite values")
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("RTC overlap weights must lie in [0, 1]")
        model_pair = (self.model_hard_prefix, self.model_overlap_target)
        if (model_pair[0] is None) != (model_pair[1] is None):
            raise ShapeMismatchError(
                "RTC model_hard_prefix and model_overlap_target must be paired"
            )
        if model_pair[0] is not None and model_pair[1] is not None:
            model_hard = np.asarray(model_pair[0], dtype=np.float32)
            model_target = np.asarray(model_pair[1], dtype=np.float32)
            if (
                model_hard.ndim != 2
                or model_target.ndim != 2
                or model_hard.shape[0] != hard.shape[0]
                or model_target.shape[0] != target.shape[0]
                or model_hard.shape[1] != model_target.shape[1]
            ):
                raise ShapeMismatchError(
                    "RTC model-space targets must be aligned [T, A_model]"
                )
            if not np.all(np.isfinite(model_hard)) or not np.all(
                np.isfinite(model_target)
            ):
                raise ShapeMismatchError("RTC model-space targets must be finite")
            object.__setattr__(self, "model_hard_prefix", model_hard)
            object.__setattr__(self, "model_overlap_target", model_target)
        object.__setattr__(self, "hard_prefix", hard)
        object.__setattr__(self, "overlap_target", target)
        object.__setattr__(self, "overlap_weights", weights)


def build_rtc_conditioning(
    previous: ActionChunk,
    executed_steps: int,
    config: RTCConfig,
) -> RTCConditioning:
    """Build conditioning from the still-relevant portion of an old chunk."""

    if not 0 <= executed_steps <= previous.horizon:
        raise ValueError("executed_steps is outside the previous action chunk")
    remaining = previous.values[executed_steps:]
    hard_count = min(
        max(0, previous.committed_prefix - executed_steps), remaining.shape[0]
    )
    hard_prefix = remaining[:hard_count].copy()
    overlap = remaining[: config.overlap_horizon].copy()
    positions = np.arange(overlap.shape[0], dtype=np.float32)
    weights = np.clip(
        config.guidance_weight * np.power(config.guidance_decay, positions),
        0.0,
        1.0,
    ).astype(np.float32)
    if hard_count:
        weights[:hard_count] = 1.0
    model_hard_prefix = None
    model_overlap = None
    if previous.model_values is not None:
        model_remaining = previous.model_values[executed_steps:]
        model_hard_prefix = model_remaining[:hard_count].copy()
        model_overlap = model_remaining[: config.overlap_horizon].copy()
    return RTCConditioning(
        hard_prefix,
        overlap,
        weights,
        model_hard_prefix=model_hard_prefix,
        model_overlap_target=model_overlap,
    )


def apply_rtc_conditioning(
    candidate: ActionChunk, conditioning: RTCConditioning
) -> ActionChunk:
    """Apply an explicitly non-native, output-space compatibility fallback.

    This function is useful for wiring and dry runs, but it cannot be cited as
    denoising-time RTC guidance: no VJP is evaluated and the model has already
    produced the complete chunk before this blend happens.
    """

    values = candidate.values.copy()
    if conditioning.overlap_target.shape[1:] != values.shape[1:]:
        raise ShapeMismatchError("RTC candidate action dimension differs from overlap")
    overlap = min(conditioning.overlap_target.shape[0], values.shape[0])
    if overlap:
        weight = conditioning.overlap_weights[:overlap, None]
        values[:overlap] = (
            weight * conditioning.overlap_target[:overlap]
            + (1.0 - weight) * values[:overlap]
        )
    hard = min(conditioning.hard_prefix.shape[0], values.shape[0])
    if hard:
        values[:hard] = conditioning.hard_prefix[:hard]
    return ActionChunk(
        values=values,
        period_s=candidate.period_s,
        source_observation_ns=candidate.source_observation_ns,
        model_values=None,
        committed_prefix=hard,
        metadata={
            **candidate.metadata,
            "rtc_postprocess_fallback": True,
            "rtc_host_postprocess_fallback": True,
            "rtc_native_guidance": False,
            "rtc_guidance_mode": "host_output_postprocess",
        },
    )


class NativeRTCPolicy(Protocol):
    """Model hook that applies constraints inside its denoising loop.

    Implementations should call :func:`rtc_torch.guided_flow_denoise` or an
    equivalent native backend. Merely calling :func:`apply_rtc_conditioning`
    after ordinary prediction does not satisfy this protocol's semantics.
    """

    def predict_with_rtc(
        self, observation: Observation, conditioning: RTCConditioning
    ) -> PolicyOutput:
        """Predict with hard masks and overlap guidance inside denoising."""


@dataclass(frozen=True)
class _PlanRequest:
    request_id: int
    observation: Observation
    conditioning: RTCConditioning | None
    submitted_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class PlannedChunk:
    """Asynchronous result with queue and model latency kept separate."""

    request_id: int
    output: PolicyOutput
    queue_wait_ms: float


@dataclass(frozen=True)
class _PlanFailure:
    request_id: int
    error: Exception


class AsynchronousChunkPlanner:
    """Generate the next chunk while the current chunk is executing."""

    def __init__(
        self,
        policy: ChunkPolicy,
        config: RTCConfig,
        *,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.metrics = metrics or RuntimeMetrics()
        self._requests: queue.Queue[_PlanRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[PlannedChunk | _PlanFailure] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name="rtc-planner", daemon=True
        )
        self._thread.start()

    def _put_latest(self, destination: queue.Queue, value: Any) -> None:
        try:
            destination.put_nowait(value)
        except queue.Full:
            try:
                destination.get_nowait()
            except queue.Empty:
                pass
            destination.put_nowait(value)

    def submit(
        self,
        request_id: int,
        observation: Observation,
        previous: ActionChunk | None = None,
        executed_steps: int = 0,
    ) -> None:
        """Submit the newest observation, dropping an obsolete queued request."""

        if self._closed.is_set():
            raise RuntimeError("RTC planner is closed")
        conditioning = (
            build_rtc_conditioning(previous, executed_steps, self.config)
            if previous is not None
            else None
        )
        self._put_latest(
            self._requests,
            _PlanRequest(
                request_id=request_id,
                observation=observation,
                conditioning=conditioning,
            ),
        )

    def _worker(self) -> None:
        while not self._closed.is_set():
            request = self._requests.get()
            if request is None:
                return
            queue_wait_ms = (time.monotonic_ns() - request.submitted_ns) / 1_000_000.0
            try:
                has_native_method = hasattr(self.policy, "predict_with_rtc")
                native_supported = bool(
                    getattr(self.policy, "rtc_native_supported", has_native_method)
                )
                if (
                    request.conditioning is not None
                    and has_native_method
                    and native_supported
                ):
                    native_output = self.policy.predict_with_rtc(
                        request.observation, request.conditioning
                    )
                    hard_length = request.conditioning.hard_prefix.shape[0]
                    native_action = native_output.action
                    action = ActionChunk(
                        values=native_action.values,
                        period_s=native_action.period_s,
                        source_observation_ns=native_action.source_observation_ns,
                        generated_ns=native_action.generated_ns,
                        model_values=native_action.model_values,
                        committed_prefix=hard_length,
                        metadata=native_action.metadata,
                    )
                    output = PolicyOutput(
                        action=action,
                        model_latency_ms=native_output.model_latency_ms,
                        path=native_output.path,
                        accepted_prefix=native_output.accepted_prefix,
                        diagnostics={
                            **native_output.diagnostics,
                            "rtc_native_guidance": True,
                            "rtc_host_postprocess_fallback": False,
                            "rtc_guidance_mode": "native_denoise_vjp",
                        },
                    )
                else:
                    output = self.policy.predict(request.observation)
                    if request.conditioning is not None:
                        action = apply_rtc_conditioning(
                            output.action, request.conditioning
                        )
                        output = PolicyOutput(
                            action=action,
                            model_latency_ms=output.model_latency_ms,
                            path=f"{output.path}+rtc_host_postprocess_fallback",
                            accepted_prefix=output.accepted_prefix,
                            diagnostics={
                                **output.diagnostics,
                                "rtc_native_guidance": False,
                                "rtc_host_postprocess_fallback": True,
                                "rtc_guidance_mode": "host_output_postprocess",
                                "rtc_native_policy_supported": native_supported,
                            },
                        )
                self.metrics.record(
                    RuntimeMetrics.MODEL_LATENCY, output.model_latency_ms
                )
                self.metrics.record("rtc/planner_queue_wait_ms", queue_wait_ms)
                result: PlannedChunk | _PlanFailure = PlannedChunk(
                    request.request_id, output, queue_wait_ms
                )
            except Exception as error:
                result = _PlanFailure(request.request_id, error)
            self._put_latest(self._results, result)

    def result(self, timeout_s: float | None = None) -> PlannedChunk:
        """Wait for a generated chunk without reclassifying model latency."""

        timeout = self.config.planner_timeout_s if timeout_s is None else timeout_s
        start = time.perf_counter()
        result = self._results.get(timeout=timeout)
        blocking_ms = (time.perf_counter() - start) * 1_000.0
        self.metrics.record(RuntimeMetrics.ROBOT_WAIT, blocking_ms)
        if isinstance(result, _PlanFailure):
            raise RuntimeError(
                f"RTC planning request {result.request_id} failed"
            ) from result.error
        return result

    def close(self) -> None:
        """Stop the worker or report that process-level recovery is required.

        Python cannot safely cancel an in-flight accelerator forward.  A
        timeout is therefore an explicit teardown failure, not a successful
        close; callers should terminate/restart the owning process before
        reclaiming that model or GPU context.
        """

        if self._closed.is_set() and not self._thread.is_alive():
            return
        self._closed.set()
        self._put_latest(self._requests, None)
        self._thread.join(timeout=self.config.planner_shutdown_timeout_s)
        if self._thread.is_alive():
            raise RuntimeError(
                "RTC planner worker is still executing after shutdown timeout; "
                "restart the owning process before reusing its model/GPU context"
            )

    def __enter__(self) -> "AsynchronousChunkPlanner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
