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

"""Draft-and-parallel-verify inference for continuous action chunks."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..config import SpeculativeConfig
from ..contracts import ActionChunk, ChunkPolicy, Observation, PolicyOutput
from ..errors import ShapeMismatchError
from ..metrics import RuntimeMetrics


@dataclass(frozen=True)
class GripperGuardResult:
    """Capability-gated gripper crossing evidence from parallel verification."""

    capability_enabled: bool
    capability_reason: str
    gripper_index: int | None = None
    switch_threshold: float | None = None
    previous_value: float | None = None
    verify_stop: bool = False
    crossing_ks: tuple[int, ...] = ()
    first_crossing_step: int | None = None

    def __post_init__(self) -> None:
        if not self.capability_reason:
            raise ValueError("gripper capability_reason must not be empty")
        if self.gripper_index is not None and self.gripper_index < 0:
            raise ValueError("gripper_index must be non-negative")
        if self.switch_threshold is not None and not np.isfinite(self.switch_threshold):
            raise ValueError("gripper switch threshold must be finite")
        if self.previous_value is not None and not np.isfinite(self.previous_value):
            raise ValueError("previous gripper value must be finite")
        if self.capability_enabled:
            if (
                self.gripper_index is None
                or self.switch_threshold is None
                or self.previous_value is None
            ):
                raise ValueError(
                    "enabled gripper capability requires index, threshold, and previous value"
                )
        elif (
            self.verify_stop or self.crossing_ks or self.first_crossing_step is not None
        ):
            raise ValueError("disabled gripper capability cannot report a crossing")
        if any(index < 0 for index in self.crossing_ks):
            raise ValueError("gripper crossing K indices must be non-negative")
        if self.first_crossing_step is not None and self.first_crossing_step < 0:
            raise ValueError("first gripper crossing step must be non-negative")
        if self.capability_enabled and self.verify_stop != bool(self.crossing_ks):
            raise ValueError("gripper verify_stop and crossing_ks disagree")
        if self.capability_enabled and self.verify_stop != (
            self.first_crossing_step is not None
        ):
            raise ValueError("gripper verify_stop and first crossing step disagree")

    @classmethod
    def disabled(
        cls,
        reason: str,
        *,
        gripper_index: int | None = None,
        switch_threshold: float | None = None,
        previous_value: float | None = None,
    ) -> "GripperGuardResult":
        """Build an explicit capability-gated disabled result."""

        return cls(
            capability_enabled=False,
            capability_reason=reason,
            gripper_index=gripper_index,
            switch_threshold=switch_threshold,
            previous_value=previous_value,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return stable diagnostics shared by verifier and policy paths."""

        return {
            "gripper_capability_enabled": self.capability_enabled,
            "gripper_capability_reason": self.capability_reason,
            "gripper_index": self.gripper_index,
            "gripper_switch_threshold": self.switch_threshold,
            "gripper_previous_value": self.previous_value,
            "gripper_verify_stop": self.verify_stop,
            "gripper_verify_crossing_ks": self.crossing_ks,
            "gripper_verify_first_crossing_step": self.first_crossing_step,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Parallel main-model clean candidates reduced to one tail candidate.

    Error tensors are either ``[T]`` for a legacy/single verifier point or
    ``[K, T]`` for K parallel flow-verification points.  Keeping the K axis is
    important: acceptance is decided independently for every verifier point
    and then reduced with ``min(prefix_len_k)``.  Averaging errors before the
    prefix decision would be more permissive than the audited FLASH runtime.
    """

    main_candidate: NDArray[np.float32]
    absolute_error: NDArray[np.float32]
    relative_error: NDArray[np.float32]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    gripper_guard: GripperGuardResult | None = None

    def __post_init__(self) -> None:
        candidate = np.asarray(self.main_candidate, dtype=np.float32)
        absolute = np.asarray(self.absolute_error, dtype=np.float32)
        relative = np.asarray(self.relative_error, dtype=np.float32)
        if candidate.ndim != 2:
            raise ShapeMismatchError("main_candidate must be [T, A]")
        if absolute.ndim not in (1, 2) or absolute.shape[-1] != candidate.shape[0]:
            raise ShapeMismatchError(
                "verification errors must be [T] or [K, T] with one value per action"
            )
        if relative.shape != absolute.shape:
            raise ShapeMismatchError("absolute and relative verification errors differ")
        if np.any(np.isnan(absolute)) or np.any(np.isnan(relative)):
            raise ValueError("verification errors must not contain NaN")
        if np.any(absolute < 0) or np.any(relative < 0):
            raise ValueError("verification errors must be non-negative")
        if self.gripper_guard is not None and any(
            index >= (1 if absolute.ndim == 1 else absolute.shape[0])
            for index in self.gripper_guard.crossing_ks
        ):
            raise ShapeMismatchError("gripper crossing K index exceeds verifier batch")
        object.__setattr__(self, "main_candidate", candidate)
        object.__setattr__(self, "absolute_error", absolute)
        object.__setattr__(self, "relative_error", relative)

    @property
    def verification_count(self) -> int:
        """Number of independently judged flow verification points."""

        return 1 if self.absolute_error.ndim == 1 else int(self.absolute_error.shape[0])

    def error_matrices(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return errors in canonical ``[K, T]`` form."""

        absolute = self.absolute_error
        relative = self.relative_error
        if absolute.ndim == 1:
            absolute = absolute[None, :]
            relative = relative[None, :]
        return absolute, relative


class ParallelVerifier(Protocol):
    """Main-model verifier contract."""

    def verify(
        self, observation: Observation, draft: ActionChunk
    ) -> VerificationResult:
        """Verify every position in one model call or batched call."""


class InterpolatedFlowVerifier:
    """Verify a draft along a batched interpolation path.

    The flow convention matches the audited realtime-vla-flash implementation:

    ``x_t = t * noise + (1 - t) * x0_draft``
    ``x0_hat = x_t - t * v_theta(x_t, t)``

    ``parallel_velocity_fn`` receives the flattened ``B*K`` verification
    batch.  This policy API carries one observation/chunk at a time, so
    ``B=1`` and its concrete input shape is ``[K, T, A]``; integrations that
    batch requests can concatenate those leading K blocks without changing
    the math.  The returned main-model flow must match that shape. Production
    π0.5 integration should provide the exact denoising seed through
    ``noise_provider`` instead of relying on the explicit zero-seed test
    fallback.
    """

    def __init__(
        self,
        parallel_velocity_fn: Callable[
            [Observation, NDArray[np.float32], NDArray[np.float32]],
            NDArray[np.float32],
        ],
        *,
        validation_steps: int | None = None,
        verification_times: Sequence[float] | None = None,
        noise_provider: Callable[[Observation, ActionChunk], NDArray[np.float32]]
        | None = None,
        distance_dims: int | None = None,
        gripper_index: int | None = None,
        gripper_switch_threshold: float | None = None,
        previous_gripper_provider: Callable[[Observation, ActionChunk], float | None]
        | None = None,
        exclude_gripper_from_radius: bool = True,
        evaluation_horizon: int | None = None,
    ) -> None:
        if validation_steps is not None and verification_times is not None:
            raise ValueError("set validation_steps or verification_times, not both")
        if verification_times is None:
            if validation_steps is None:
                times = np.asarray((0.10, 0.05), dtype=np.float32)
            else:
                if validation_steps < 2:
                    raise ValueError("validation_steps must be at least two")
                times = np.linspace(0.10, 0.05, validation_steps, dtype=np.float32)
        else:
            times = np.asarray(tuple(verification_times), dtype=np.float32)
            if times.ndim != 1 or not times.size:
                raise ValueError("verification_times must be a non-empty sequence")
        if not np.all(np.isfinite(times)) or np.any(times <= 0) or np.any(times > 1):
            raise ValueError("verification times must be finite and lie in (0, 1]")
        if distance_dims is not None and distance_dims <= 0:
            raise ValueError("distance_dims must be positive when provided")
        if gripper_index is not None and gripper_index < 0:
            raise ValueError("gripper_index must be non-negative when provided")
        if gripper_switch_threshold is not None and not np.isfinite(
            gripper_switch_threshold
        ):
            raise ValueError("gripper_switch_threshold must be finite when provided")
        if evaluation_horizon is not None and evaluation_horizon <= 0:
            raise ValueError("evaluation_horizon must be positive when provided")
        self.parallel_velocity_fn = parallel_velocity_fn
        self.verification_times = times
        self.validation_steps = int(times.size)
        self.noise_provider = noise_provider
        self.distance_dims = distance_dims
        self.gripper_index = gripper_index
        self.gripper_switch_threshold = gripper_switch_threshold
        self.previous_gripper_provider = previous_gripper_provider
        self.exclude_gripper_from_radius = bool(exclude_gripper_from_radius)
        self.evaluation_horizon = evaluation_horizon

    def _gripper_guard(
        self,
        observation: Observation,
        draft: ActionChunk,
        x0_hat: NDArray[np.float32],
        evaluated_horizon: int,
    ) -> GripperGuardResult:
        configured = (
            self.gripper_index is not None,
            self.gripper_switch_threshold is not None,
            self.previous_gripper_provider is not None,
        )
        if not any(configured):
            return GripperGuardResult.disabled("not_configured")
        if not all(configured):
            return GripperGuardResult.disabled(
                "incomplete_configuration",
                gripper_index=self.gripper_index,
                switch_threshold=self.gripper_switch_threshold,
            )

        assert self.gripper_index is not None
        assert self.gripper_switch_threshold is not None
        assert self.previous_gripper_provider is not None
        if self.gripper_index >= draft.action_dim:
            return GripperGuardResult.disabled(
                "index_out_of_range",
                gripper_index=self.gripper_index,
                switch_threshold=self.gripper_switch_threshold,
            )

        previous_raw = self.previous_gripper_provider(observation, draft)
        if previous_raw is None:
            return GripperGuardResult.disabled(
                "previous_value_unavailable",
                gripper_index=self.gripper_index,
                switch_threshold=self.gripper_switch_threshold,
            )
        previous_array = np.asarray(previous_raw)
        if previous_array.ndim != 0:
            return GripperGuardResult.disabled(
                "previous_value_not_scalar",
                gripper_index=self.gripper_index,
                switch_threshold=self.gripper_switch_threshold,
            )
        previous_value = float(previous_array)
        if not np.isfinite(previous_value):
            return GripperGuardResult.disabled(
                "previous_value_not_finite",
                gripper_index=self.gripper_index,
                switch_threshold=self.gripper_switch_threshold,
            )

        values = x0_hat[:, :evaluated_horizon, self.gripper_index]
        previous = np.concatenate(
            (
                np.full((values.shape[0], 1), previous_value, dtype=np.float32),
                values[:, :-1],
            ),
            axis=1,
        )
        threshold = float(self.gripper_switch_threshold)
        crossing = ((previous < threshold) & (values >= threshold)) | (
            (previous >= threshold) & (values < threshold)
        )
        crossing_ks = tuple(
            int(index) for index in np.flatnonzero(crossing.any(axis=1))
        )
        first_crossing_step = (
            int(np.argwhere(crossing)[:, 1].min()) if crossing_ks else None
        )
        return GripperGuardResult(
            capability_enabled=True,
            capability_reason="enabled",
            gripper_index=self.gripper_index,
            switch_threshold=threshold,
            previous_value=previous_value,
            verify_stop=bool(crossing_ks),
            crossing_ks=crossing_ks,
            first_crossing_step=first_crossing_step,
        )

    def verify(
        self, observation: Observation, draft: ActionChunk
    ) -> VerificationResult:
        """Run one batched flow call and score every action position."""

        if self.noise_provider is None:
            noise = np.zeros_like(draft.values)
        else:
            noise = np.asarray(
                self.noise_provider(observation, draft), dtype=np.float32
            )
        if noise.shape != draft.values.shape:
            raise ShapeMismatchError("draft denoising seed must match its action chunk")
        times = self.verification_times.copy()
        # B=1 here.  The leading dimension is the flattened B*K verifier batch.
        states = (
            times[:, None, None] * noise[None, :, :]
            + (1.0 - times[:, None, None]) * draft.values[None, :, :]
        )
        start_ns = time.perf_counter_ns()
        predicted_velocity = np.asarray(
            self.parallel_velocity_fn(observation, states, times), dtype=np.float32
        )
        verification_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        if predicted_velocity.shape != states.shape:
            raise ShapeMismatchError(
                "parallel main-model velocity output must match interpolation states"
            )
        x0_hat = states - times[:, None, None] * predicted_velocity

        evaluated_horizon = draft.horizon
        if self.evaluation_horizon is not None:
            evaluated_horizon = min(draft.horizon, int(self.evaluation_horizon))
        gripper_guard = self._gripper_guard(
            observation, draft, x0_hat, evaluated_horizon
        )

        action_dim = int(draft.values.shape[1])
        radius_dim_limit = (
            action_dim
            if self.distance_dims is None
            else min(action_dim, int(self.distance_dims))
        )
        distance_indices = np.arange(radius_dim_limit, dtype=np.int64)
        # A gripper coordinate is excluded only when the complete crossing
        # capability is live. There is deliberately no implicit "dim 6" rule.
        if (
            self.exclude_gripper_from_radius
            and gripper_guard.capability_enabled
            and gripper_guard.gripper_index is not None
        ):
            distance_indices = distance_indices[
                distance_indices != gripper_guard.gripper_index
            ]
        eval_dims = int(distance_indices.size)
        if not eval_dims:
            raise ShapeMismatchError("no action dimensions remain for verification")

        difference = (
            x0_hat[:, :, distance_indices] - draft.values[None, :, distance_indices]
        )
        normalizer = np.sqrt(np.float32(eval_dims))
        absolute = np.linalg.norm(difference, axis=-1) / normalizer
        draft_scale = (
            np.linalg.norm(draft.values[:, distance_indices], axis=-1) / normalizer
        ).clip(min=1e-6)
        relative = absolute / draft_scale[None, :]

        if self.evaluation_horizon is not None:
            absolute[:, evaluated_horizon:] = np.inf
            relative[:, evaluated_horizon:] = np.inf

        # As in FLASH, K is reduced only for the rejected tail.  Acceptance
        # below still sees the unreduced [K,T] errors.
        main_candidate = x0_hat.mean(axis=0)
        return VerificationResult(
            main_candidate=main_candidate,
            absolute_error=absolute,
            relative_error=relative,
            diagnostics={
                "verification_ms": verification_ms,
                "validation_steps": self.validation_steps,
                "verification_times": tuple(float(value) for value in times),
                "batch_size": 1,
                "parallel_verify_batch": self.validation_steps,
                "flow_convention": "x_t=t*noise+(1-t)*draft; x0=x_t-t*v",
                "distance_dims": eval_dims,
                "distance_indices": tuple(int(index) for index in distance_indices),
                "evaluation_horizon": evaluated_horizon,
                "acceptance_reduction": "per_k_prefix_then_min",
                "tail_reduction": "mean_k_x0_hat",
                "zero_seed_fallback": self.noise_provider is None,
                **gripper_guard.diagnostics(),
            },
            gripper_guard=gripper_guard,
        )


class SpeculativeChunkPolicy:
    """Accept only a contiguous compliant draft prefix, else use main policy."""

    def __init__(
        self,
        draft_policy: ChunkPolicy,
        main_policy: ChunkPolicy,
        verifier: ParallelVerifier,
        config: SpeculativeConfig,
        *,
        metrics: RuntimeMetrics | None = None,
        resume_override: bool = True,
    ) -> None:
        config.validate()
        self.draft_policy = draft_policy
        self.main_policy = main_policy
        self.verifier = verifier
        self.config = config
        self.metrics = metrics or RuntimeMetrics()
        # Upstream FLASH schedules a full-model round for the *next* call.
        # The resume requires immediate same-round takeover; keep that behavior
        # explicit instead of silently changing the upstream state machine.
        self.resume_override = bool(resume_override)
        self._fallback_cooldown = 0
        self._pending_upstream_full = False
        self._pending_upstream_reason: str | None = None

    def _prefix_lengths(self, result: VerificationResult) -> NDArray[np.int64]:
        absolute, relative = result.error_matrices()
        compliant = (absolute <= self.config.absolute_error_threshold) | (
            relative <= self.config.relative_error_threshold
        )
        prefix_mask = np.cumprod(compliant.astype(np.int64), axis=1)
        return prefix_mask.sum(axis=1, dtype=np.int64)

    def _accepted_prefix(self, result: VerificationResult) -> int:
        """Return ``min(prefix_len_k)`` without averaging K before judging."""

        return int(self._prefix_lengths(result).min())

    @staticmethod
    def _stitch(
        draft: ActionChunk, verification: VerificationResult, prefix: int
    ) -> NDArray[np.float32]:
        if verification.main_candidate.shape != draft.values.shape:
            raise ShapeMismatchError("main candidate and draft chunk shapes differ")
        merged = verification.main_candidate.copy()
        merged[:prefix] = draft.values[:prefix]
        return merged

    @classmethod
    def _apply_gripper_guards(
        cls,
        draft: ActionChunk,
        verification: VerificationResult,
        radius_prefix: int,
    ) -> tuple[int, NDArray[np.float32], dict[str, Any]]:
        guard = verification.gripper_guard or GripperGuardResult.disabled(
            "verifier_did_not_report_capability"
        )
        if (
            guard.capability_enabled
            and guard.gripper_index is not None
            and guard.gripper_index >= draft.action_dim
        ):
            guard = GripperGuardResult.disabled(
                "index_out_of_range",
                gripper_index=guard.gripper_index,
                switch_threshold=guard.switch_threshold,
                previous_value=guard.previous_value,
            )

        verify_stop = guard.capability_enabled and guard.verify_stop
        effective_prefix = 0 if verify_stop else radius_prefix
        merged = cls._stitch(draft, verification, effective_prefix)
        post_verify_cut = False
        post_verify_cut_index: int | None = None

        if guard.capability_enabled and not verify_stop and effective_prefix > 0:
            assert guard.gripper_index is not None
            assert guard.switch_threshold is not None
            assert guard.previous_value is not None
            values = merged[:effective_prefix, guard.gripper_index]
            previous = np.concatenate(
                (
                    np.asarray([guard.previous_value], dtype=np.float32),
                    values[:-1],
                )
            )
            threshold = float(guard.switch_threshold)
            crossing = ((previous < threshold) & (values >= threshold)) | (
                (previous >= threshold) & (values < threshold)
            )
            crossing_steps = np.flatnonzero(crossing)
            if crossing_steps.size:
                post_verify_cut = True
                post_verify_cut_index = int(crossing_steps[0])
                effective_prefix = post_verify_cut_index
                # Re-stitch after shortening the prefix. This prevents draft
                # actions after the crossing from leaking into the tail.
                merged = cls._stitch(draft, verification, effective_prefix)

        diagnostics = {
            **guard.diagnostics(),
            "gripper_post_verify_cut": post_verify_cut,
            "gripper_post_verify_cut_index": post_verify_cut_index,
            "gripper_radius_prefix_before_guard": radius_prefix,
            "gripper_effective_prefix": effective_prefix,
            "gripper_fallback_triggered": bool(verify_stop or post_verify_cut),
        }
        return effective_prefix, merged, diagnostics

    @staticmethod
    def _chunk_from_draft(
        observation: Observation,
        draft_output: PolicyOutput,
        values: NDArray[np.float32],
    ) -> ActionChunk:
        return ActionChunk(
            values=values,
            period_s=draft_output.action.period_s,
            source_observation_ns=observation.timestamp_ns,
            metadata={
                **draft_output.action.metadata,
                "draft_path": draft_output.path,
            },
        )

    def predict(self, observation: Observation) -> PolicyOutput:
        """Draft, parallel-verify, and selectively fall back to main denoising."""

        if self._pending_upstream_full:
            self._pending_upstream_full = False
            pending_reason = self._pending_upstream_reason
            self._pending_upstream_reason = None
            output = self.main_policy.predict(observation)
            self.metrics.increment("speculative/upstream_next_round_main")
            return PolicyOutput(
                action=output.action,
                model_latency_ms=output.model_latency_ms,
                path="fallback_main_upstream_next_round",
                accepted_prefix=0,
                diagnostics={
                    **output.diagnostics,
                    "resume_override": False,
                    "fallback_semantics": "upstream_next_round_full",
                    "upstream_reference_fallback": "next_round_pending_full",
                    "differs_from_upstream_next_round": False,
                    "triggered_by_previous_round": True,
                    "previous_round_fallback_reason": pending_reason,
                    "scheduled_full_fallback": False,
                    "used_full_fallback": True,
                },
            )

        if not self.config.enabled or self._fallback_cooldown > 0:
            cooldown_active = self._fallback_cooldown > 0
            self._fallback_cooldown = max(0, self._fallback_cooldown - 1)
            output = self.main_policy.predict(observation)
            self.metrics.increment("speculative/cooldown_main")
            return PolicyOutput(
                action=output.action,
                model_latency_ms=output.model_latency_ms,
                path="fallback_main_cooldown",
                accepted_prefix=0,
                diagnostics={
                    **output.diagnostics,
                    "resume_override": self.resume_override,
                    "fallback_semantics": (
                        "resume_override_cooldown_full"
                        if cooldown_active
                        else "speculative_disabled_full"
                    ),
                    "upstream_reference_fallback": "next_round_pending_full",
                    "differs_from_upstream_next_round": cooldown_active,
                    "scheduled_full_fallback": False,
                    "used_full_fallback": True,
                    "cooldown_remaining": self._fallback_cooldown,
                },
            )

        start_ns = time.perf_counter_ns()
        draft_output = self.draft_policy.predict(observation)
        verification = self.verifier.verify(observation, draft_output.action)
        prefix_lengths = self._prefix_lengths(verification)
        radius_prefix = int(prefix_lengths.min())
        prefix, merged, gripper_diagnostics = self._apply_gripper_guards(
            draft_output.action, verification, radius_prefix
        )
        horizon = draft_output.action.horizon
        acceptance_ratio = prefix / horizon
        low_acceptance = prefix < self.config.minimum_prefix or (
            acceptance_ratio < self.config.fallback_acceptance_ratio
        )
        gripper_fallback = bool(gripper_diagnostics["gripper_fallback_triggered"])
        if low_acceptance or gripper_fallback:
            self.metrics.record("speculative/acceptance_ratio", acceptance_ratio)
            if gripper_diagnostics["gripper_verify_stop"]:
                fallback_reason = "gripper_verify_stop"
            elif gripper_diagnostics["gripper_post_verify_cut"]:
                fallback_reason = "gripper_post_verify_cut"
            else:
                fallback_reason = "low_acceptance"
            common_diagnostics = {
                **verification.diagnostics,
                **gripper_diagnostics,
                "acceptance_ratio": acceptance_ratio,
                "prefix_lengths_per_verify": tuple(
                    int(value) for value in prefix_lengths
                ),
                "min_k_prefix": prefix,
                "radius_min_k_prefix": radius_prefix,
                "draft_model_ms": draft_output.model_latency_ms,
                "upstream_reference_fallback": "next_round_pending_full",
                "acceptance_rule": "absolute_or_relative_threshold",
                "fallback_reason": fallback_reason,
            }
            if self.resume_override:
                self._fallback_cooldown = self.config.fallback_cooldown_chunks
                main_output = self.main_policy.predict(observation)
                total_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                self.metrics.increment("speculative/fallback_main")
                return PolicyOutput(
                    action=main_output.action,
                    model_latency_ms=total_ms,
                    path="fallback_main_low_acceptance",
                    accepted_prefix=prefix,
                    diagnostics={
                        **common_diagnostics,
                        "main_model_ms": main_output.model_latency_ms,
                        "resume_override": True,
                        "fallback_semantics": "resume_override_same_round_full",
                        "differs_from_upstream_next_round": True,
                        "scheduled_full_fallback": False,
                        "used_full_fallback": True,
                        "resume_override_cooldown_chunks": (
                            self.config.fallback_cooldown_chunks
                        ),
                    },
                )

            # Audited upstream behavior: return this round's verifier-tail
            # stitch, then force a full main-model round on the next call.
            self._pending_upstream_full = True
            self._pending_upstream_reason = fallback_reason
            total_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            self.metrics.increment("speculative/upstream_fallback_scheduled")
            return PolicyOutput(
                action=self._chunk_from_draft(observation, draft_output, merged),
                model_latency_ms=total_ms,
                path="speculative_low_acceptance_pending_full",
                accepted_prefix=prefix,
                diagnostics={
                    **common_diagnostics,
                    "resume_override": False,
                    "fallback_semantics": "upstream_next_round_pending",
                    "differs_from_upstream_next_round": False,
                    "main_model_ms": 0.0,
                    "scheduled_full_fallback": True,
                    "used_full_fallback": False,
                },
            )

        total_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        self.metrics.increment("speculative/accepted")
        self.metrics.record("speculative/acceptance_ratio", acceptance_ratio)
        return PolicyOutput(
            action=self._chunk_from_draft(observation, draft_output, merged),
            model_latency_ms=total_ms,
            path="speculative_parallel_verify",
            accepted_prefix=prefix,
            diagnostics={
                **verification.diagnostics,
                **gripper_diagnostics,
                "acceptance_ratio": acceptance_ratio,
                "prefix_lengths_per_verify": tuple(
                    int(value) for value in prefix_lengths
                ),
                "min_k_prefix": prefix,
                "radius_min_k_prefix": radius_prefix,
                "draft_model_ms": draft_output.model_latency_ms,
                "resume_override": self.resume_override,
                "fallback_semantics": "not_triggered",
                "upstream_reference_fallback": "next_round_pending_full",
                "acceptance_rule": "absolute_or_relative_threshold",
                "scheduled_full_fallback": False,
                "used_full_fallback": False,
            },
        )
