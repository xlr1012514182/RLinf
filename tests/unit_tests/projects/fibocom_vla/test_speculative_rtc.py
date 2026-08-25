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

"""Unit tests for continuous speculative verification and native RTC guidance."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import RTCConfig, SpeculativeConfig
from rlinf.projects.fibocom_vla.contracts import (
    ActionChunk,
    Observation,
    PolicyOutput,
    RobotState,
)
from rlinf.projects.fibocom_vla.inference.rtc import (
    AsynchronousChunkPlanner,
    RTCConditioning,
    apply_rtc_conditioning,
    build_rtc_conditioning,
)
from rlinf.projects.fibocom_vla.inference.speculative import (
    GripperGuardResult,
    InterpolatedFlowVerifier,
    SpeculativeChunkPolicy,
    VerificationResult,
)


def _observation() -> Observation:
    return Observation(
        state=RobotState(
            joint_positions=np.zeros(2, dtype=np.float32),
            joint_names=("joint_0", "joint_1"),
        ),
        images={"main": np.zeros((4, 4, 3), dtype=np.uint8)},
        instruction="stack the blocks",
        timestamp_ns=123,
    )


class _StaticPolicy:
    def __init__(self, values: np.ndarray, *, path: str) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.path = path
        self.calls = 0

    def predict(self, observation: Observation) -> PolicyOutput:
        self.calls += 1
        return PolicyOutput(
            action=ActionChunk(
                values=self.values.copy(),
                period_s=0.05,
                source_observation_ns=observation.timestamp_ns,
            ),
            model_latency_ms=0.25,
            path=self.path,
        )


class _StaticVerifier:
    def __init__(self, result: VerificationResult) -> None:
        self.result = result

    def verify(
        self, observation: Observation, draft: ActionChunk
    ) -> VerificationResult:
        del observation, draft
        return self.result


def _spec_config(*, minimum_prefix: int, fallback_ratio: float) -> SpeculativeConfig:
    return SpeculativeConfig(
        relative_error_threshold=0.5,
        absolute_error_threshold=0.5,
        minimum_prefix=minimum_prefix,
        fallback_acceptance_ratio=fallback_ratio,
        fallback_cooldown_chunks=0,
    )


def test_flow_interpolation_direction_and_bk_clean_estimate() -> None:
    draft_values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    noise = np.asarray([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)
    times = np.asarray([0.20, 0.10], dtype=np.float32)
    x0_hat = np.stack((draft_values + 0.2, draft_values - 0.1), axis=0).astype(
        np.float32
    )
    captured: dict[str, Any] = {}

    def velocity_fn(
        observation: Observation,
        states: np.ndarray,
        got_times: np.ndarray,
    ) -> np.ndarray:
        del observation
        captured["states"] = states.copy()
        captured["times"] = got_times.copy()
        # x0_hat = x_t - t*v  ->  v = (x_t - x0_hat) / t
        return (states - x0_hat) / got_times[:, None, None]

    verifier = InterpolatedFlowVerifier(
        velocity_fn,
        verification_times=times,
        noise_provider=lambda _observation, _draft: noise,
    )
    result = verifier.verify(
        _observation(),
        ActionChunk(draft_values, period_s=0.05, source_observation_ns=123),
    )

    expected_states = (
        times[:, None, None] * noise[None, :, :]
        + (1.0 - times[:, None, None]) * draft_values[None, :, :]
    )
    np.testing.assert_allclose(captured["states"], expected_states)
    np.testing.assert_allclose(captured["times"], times)
    np.testing.assert_allclose(result.main_candidate, x0_hat.mean(axis=0))
    assert result.absolute_error.shape == (2, 2)
    assert result.diagnostics["parallel_verify_batch"] == 2
    assert result.diagnostics["batch_size"] == 1


def test_no_gripper_configuration_disables_capability_without_fixed_dimension() -> None:
    draft_values = np.ones((3, 7), dtype=np.float32)
    desired = np.repeat(draft_values[None, :, :], 2, axis=0)
    desired[:, :, 6] += 1.0
    times = np.asarray([0.20, 0.10], dtype=np.float32)

    def velocity_fn(
        observation: Observation,
        states: np.ndarray,
        got_times: np.ndarray,
    ) -> np.ndarray:
        del observation
        return (states - desired) / got_times[:, None, None]

    result = InterpolatedFlowVerifier(
        velocity_fn,
        verification_times=times,
    ).verify(
        _observation(),
        ActionChunk(draft_values, period_s=0.05, source_observation_ns=123),
    )

    assert result.diagnostics["gripper_capability_enabled"] is False
    assert result.diagnostics["gripper_capability_reason"] == "not_configured"
    assert result.diagnostics["gripper_index"] is None
    assert result.diagnostics["distance_indices"] == tuple(range(7))
    # The changed seventh coordinate participates in the radius: there is no
    # implicit assumption that index 6 is a gripper.
    assert np.all(result.absolute_error > 0)

    incomplete = InterpolatedFlowVerifier(
        velocity_fn,
        verification_times=times,
        gripper_index=6,
        gripper_switch_threshold=0.5,
        # A runtime previous-value provider is deliberately absent.
    ).verify(
        _observation(),
        ActionChunk(draft_values, period_s=0.05, source_observation_ns=123),
    )
    assert incomplete.diagnostics["gripper_capability_enabled"] is False
    assert incomplete.diagnostics["gripper_capability_reason"] == (
        "incomplete_configuration"
    )
    assert incomplete.diagnostics["distance_indices"] == tuple(range(7))


def test_any_k_gripper_crossing_stops_prefix_and_resumes_main_same_round() -> None:
    draft_values = np.zeros((4, 3), dtype=np.float32)
    draft_values[:, 1] = 0.1
    desired = np.repeat(draft_values[None, :, :], 2, axis=0)
    # Only K=1 crosses from below to above the threshold, midway through the
    # execution window. K disagreement must still stop the whole prefix.
    desired[1, 2:, 1] = 0.8
    times = np.asarray([0.20, 0.10], dtype=np.float32)

    def velocity_fn(
        observation: Observation,
        states: np.ndarray,
        got_times: np.ndarray,
    ) -> np.ndarray:
        del observation
        return (states - desired) / got_times[:, None, None]

    verifier = InterpolatedFlowVerifier(
        velocity_fn,
        verification_times=times,
        gripper_index=1,
        gripper_switch_threshold=0.5,
        previous_gripper_provider=lambda _observation, _draft: 0.1,
    )
    main_values = np.full_like(draft_values, 7.0)
    main = _StaticPolicy(main_values, path="main")
    policy = SpeculativeChunkPolicy(
        _StaticPolicy(draft_values, path="draft"),
        main,
        verifier,
        _spec_config(minimum_prefix=0, fallback_ratio=0.0),
        resume_override=True,
    )

    output = policy.predict(_observation())

    np.testing.assert_allclose(output.action.values, main_values)
    assert main.calls == 1
    assert output.accepted_prefix == 0
    assert output.diagnostics["radius_min_k_prefix"] == 4
    assert output.diagnostics["gripper_effective_prefix"] == 0
    assert output.diagnostics["gripper_capability_enabled"] is True
    assert output.diagnostics["gripper_index"] == 1
    assert output.diagnostics["distance_indices"] == (0, 2)
    assert output.diagnostics["gripper_verify_stop"] is True
    assert output.diagnostics["gripper_verify_crossing_ks"] == (1,)
    assert output.diagnostics["gripper_verify_first_crossing_step"] == 2
    assert output.diagnostics["gripper_post_verify_cut"] is False
    assert output.diagnostics["gripper_fallback_triggered"] is True
    assert output.diagnostics["fallback_reason"] == "gripper_verify_stop"
    assert output.diagnostics["fallback_semantics"] == (
        "resume_override_same_round_full"
    )


def test_gripper_verify_stop_returns_tail_then_runs_upstream_full_round() -> None:
    draft_values = np.arange(12, dtype=np.float32).reshape(4, 3)
    verifier_tail = np.full_like(draft_values, -4.0)
    accepted = np.zeros((2, 4), dtype=np.float32)
    guard = GripperGuardResult(
        capability_enabled=True,
        capability_reason="enabled",
        gripper_index=2,
        switch_threshold=0.5,
        previous_value=0.1,
        verify_stop=True,
        crossing_ks=(0,),
        first_crossing_step=1,
    )
    verifier = _StaticVerifier(
        VerificationResult(
            verifier_tail,
            accepted,
            accepted.copy(),
            gripper_guard=guard,
        )
    )
    main_values = np.full_like(draft_values, 8.0)
    main = _StaticPolicy(main_values, path="main")
    policy = SpeculativeChunkPolicy(
        _StaticPolicy(draft_values, path="draft"),
        main,
        verifier,
        _spec_config(minimum_prefix=0, fallback_ratio=0.0),
        resume_override=False,
    )

    current = policy.predict(_observation())

    np.testing.assert_allclose(current.action.values, verifier_tail)
    assert current.accepted_prefix == 0
    assert main.calls == 0
    assert current.diagnostics["gripper_verify_stop"] is True
    assert current.diagnostics["gripper_post_verify_cut"] is False
    assert current.diagnostics["fallback_reason"] == "gripper_verify_stop"
    assert current.diagnostics["scheduled_full_fallback"] is True

    following = policy.predict(_observation())
    np.testing.assert_allclose(following.action.values, main_values)
    assert main.calls == 1
    assert following.diagnostics["previous_round_fallback_reason"] == (
        "gripper_verify_stop"
    )


def test_min_k_prefix_is_judged_before_tail_mean_stitch() -> None:
    draft_values = np.arange(8, dtype=np.float32).reshape(4, 2)
    main_tail = np.full_like(draft_values, 9.0)
    # K0 accepts two actions; K1 accepts only one. The result must be min=1.
    error = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    verifier = _StaticVerifier(VerificationResult(main_tail, error, error.copy()))
    draft = _StaticPolicy(draft_values, path="draft")
    main = _StaticPolicy(np.full_like(draft_values, 42.0), path="main")
    policy = SpeculativeChunkPolicy(
        draft,
        main,
        verifier,
        _spec_config(minimum_prefix=0, fallback_ratio=0.0),
    )

    output = policy.predict(_observation())

    expected = main_tail.copy()
    expected[:1] = draft_values[:1]
    np.testing.assert_allclose(output.action.values, expected)
    assert output.accepted_prefix == 1
    assert output.diagnostics["prefix_lengths_per_verify"] == (2, 1)
    assert main.calls == 0


def test_low_acceptance_resume_override_runs_main_in_same_round() -> None:
    draft_values = np.zeros((4, 2), dtype=np.float32)
    verifier_tail = np.full_like(draft_values, 3.0)
    rejected = np.ones((2, 4), dtype=np.float32)
    main_values = np.full_like(draft_values, 7.0)
    main = _StaticPolicy(main_values, path="main")
    policy = SpeculativeChunkPolicy(
        _StaticPolicy(draft_values, path="draft"),
        main,
        _StaticVerifier(VerificationResult(verifier_tail, rejected, rejected.copy())),
        _spec_config(minimum_prefix=1, fallback_ratio=0.5),
        resume_override=True,
    )

    output = policy.predict(_observation())

    np.testing.assert_allclose(output.action.values, main_values)
    assert main.calls == 1
    assert output.diagnostics["resume_override"] is True
    assert output.diagnostics["fallback_semantics"] == (
        "resume_override_same_round_full"
    )
    assert output.diagnostics["differs_from_upstream_next_round"] is True
    assert output.diagnostics["scheduled_full_fallback"] is False
    assert output.diagnostics["used_full_fallback"] is True


def test_upstream_mode_stitches_now_and_runs_main_on_next_round() -> None:
    draft_values = np.zeros((4, 2), dtype=np.float32)
    verifier_tail = np.full_like(draft_values, 3.0)
    rejected = np.ones((2, 4), dtype=np.float32)
    main_values = np.full_like(draft_values, 7.0)
    main = _StaticPolicy(main_values, path="main")
    policy = SpeculativeChunkPolicy(
        _StaticPolicy(draft_values, path="draft"),
        main,
        _StaticVerifier(VerificationResult(verifier_tail, rejected, rejected.copy())),
        _spec_config(minimum_prefix=1, fallback_ratio=0.5),
        resume_override=False,
    )

    current = policy.predict(_observation())
    np.testing.assert_allclose(current.action.values, verifier_tail)
    assert main.calls == 0
    assert current.path == "speculative_low_acceptance_pending_full"
    assert current.diagnostics["fallback_semantics"] == ("upstream_next_round_pending")
    assert current.diagnostics["scheduled_full_fallback"] is True
    assert current.diagnostics["used_full_fallback"] is False

    following = policy.predict(_observation())
    np.testing.assert_allclose(following.action.values, main_values)
    assert main.calls == 1
    assert following.path == "fallback_main_upstream_next_round"
    assert following.diagnostics["scheduled_full_fallback"] is False
    assert following.diagnostics["used_full_fallback"] is True


@pytest.mark.parametrize("crossing_step", [0, 2])
def test_post_verify_gripper_cut_restitches_tail_and_schedules_full(
    crossing_step: int,
) -> None:
    draft_values = np.arange(12, dtype=np.float32).reshape(4, 3)
    draft_values[:, 1] = 0.1
    draft_values[crossing_step:, 1] = 0.8
    verifier_tail = np.full_like(draft_values, -5.0)
    accepted = np.zeros((2, 4), dtype=np.float32)
    guard = GripperGuardResult(
        capability_enabled=True,
        capability_reason="enabled",
        gripper_index=1,
        switch_threshold=0.5,
        previous_value=0.1,
    )
    verifier = _StaticVerifier(
        VerificationResult(
            verifier_tail,
            accepted,
            accepted.copy(),
            gripper_guard=guard,
        )
    )
    main_values = np.full_like(draft_values, 9.0)
    main = _StaticPolicy(main_values, path="main")
    policy = SpeculativeChunkPolicy(
        _StaticPolicy(draft_values, path="draft"),
        main,
        verifier,
        _spec_config(minimum_prefix=0, fallback_ratio=0.0),
        resume_override=False,
    )

    current = policy.predict(_observation())

    expected = verifier_tail.copy()
    expected[:crossing_step] = draft_values[:crossing_step]
    np.testing.assert_allclose(current.action.values, expected)
    assert current.accepted_prefix == crossing_step
    assert main.calls == 0
    assert current.diagnostics["radius_min_k_prefix"] == 4
    assert current.diagnostics["gripper_verify_stop"] is False
    assert current.diagnostics["gripper_post_verify_cut"] is True
    assert current.diagnostics["gripper_post_verify_cut_index"] == crossing_step
    assert current.diagnostics["gripper_effective_prefix"] == crossing_step
    assert current.diagnostics["gripper_fallback_triggered"] is True
    assert current.diagnostics["fallback_reason"] == "gripper_post_verify_cut"
    assert current.diagnostics["scheduled_full_fallback"] is True
    assert current.diagnostics["used_full_fallback"] is False

    following = policy.predict(_observation())
    np.testing.assert_allclose(following.action.values, main_values)
    assert main.calls == 1
    assert following.diagnostics["previous_round_fallback_reason"] == (
        "gripper_post_verify_cut"
    )


def test_host_rtc_fallback_is_explicitly_not_native_guidance() -> None:
    previous = ActionChunk(
        values=np.arange(8, dtype=np.float32).reshape(4, 2),
        model_values=np.arange(12, dtype=np.float32).reshape(4, 3),
        period_s=0.05,
        source_observation_ns=1,
        committed_prefix=3,
    )
    config = RTCConfig(
        execution_horizon=2,
        overlap_horizon=3,
        guidance_weight=1.0,
        guidance_decay=0.5,
    )
    conditioning = build_rtc_conditioning(previous, executed_steps=1, config=config)
    np.testing.assert_allclose(conditioning.overlap_weights, [1.0, 1.0, 0.25])
    np.testing.assert_allclose(
        conditioning.model_hard_prefix, previous.model_values[1:3]
    )
    np.testing.assert_allclose(
        conditioning.model_overlap_target, previous.model_values[1:4]
    )

    candidate = ActionChunk(
        values=np.zeros((3, 2), dtype=np.float32),
        model_values=np.zeros((3, 3), dtype=np.float32),
        period_s=0.05,
        source_observation_ns=2,
    )
    output = apply_rtc_conditioning(candidate, conditioning)
    np.testing.assert_allclose(output.values[:2], previous.values[1:3])
    assert output.metadata["rtc_native_guidance"] is False
    assert output.metadata["rtc_host_postprocess_fallback"] is True
    assert output.metadata["rtc_guidance_mode"] == "host_output_postprocess"
    assert output.model_values is None


def test_planner_shutdown_timeout_is_an_explicit_teardown_failure() -> None:
    started = threading.Event()
    release = threading.Event()

    class _BlockingPolicy:
        def predict(self, observation: Observation) -> PolicyOutput:
            started.set()
            release.wait(timeout=1.0)
            return _StaticPolicy(np.zeros((1, 2)), path="blocking").predict(observation)

    planner = AsynchronousChunkPlanner(
        _BlockingPolicy(),
        RTCConfig(
            execution_horizon=1,
            overlap_horizon=1,
            planner_shutdown_timeout_s=0.01,
        ),
    )
    planner.submit(0, _observation())
    assert started.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="restart the owning process"):
        planner.close()
    release.set()
    for _ in range(100):
        try:
            planner.close()
            break
        except RuntimeError:
            time.sleep(0.001)
    else:
        pytest.fail("planner did not exit after releasing the blocked forward")


def test_torch_exponential_position_weights() -> None:
    torch = pytest.importorskip("torch")
    from rlinf.projects.fibocom_vla.inference.rtc_torch import (
        exponential_position_weights,
    )

    weights = exponential_position_weights(
        4, initial_weight=0.8, decay=0.5, device="cpu"
    )
    torch.testing.assert_close(
        weights, torch.tensor([0.8, 0.4, 0.2, 0.1], dtype=torch.float32)
    )


def test_torch_rtc_freezes_prefix_and_applies_vjp_inside_denoise() -> None:
    torch = pytest.importorskip("torch")
    from rlinf.projects.fibocom_vla.inference.rtc_torch import (
        guided_flow_denoise,
    )

    conditioning = RTCConditioning(
        hard_prefix=np.asarray([[0.25, -0.50]], dtype=np.float32),
        overlap_target=np.asarray(
            [[0.25, -0.50], [1.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        overlap_weights=np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
    )
    initial = torch.zeros((3, 2), dtype=torch.float32)
    seen_inputs: list[torch.Tensor] = []
    seen_requires_grad: list[bool] = []

    def zero_velocity(actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        seen_inputs.append(actions.detach().clone())
        seen_requires_grad.append(actions.requires_grad)
        return actions * 0.0

    unguided, _ = guided_flow_denoise(
        zero_velocity,
        initial,
        [1.0, 0.5, 0.0],
        conditioning,
        guidance_scale=0.0,
    )
    seen_inputs.clear()
    seen_requires_grad.clear()
    guided, diagnostics = guided_flow_denoise(
        zero_velocity,
        initial,
        [1.0, 0.5, 0.0],
        conditioning,
        guidance_scale=1.0,
    )

    expected_prefix = torch.tensor([0.25, -0.50])
    torch.testing.assert_close(guided[0], expected_prefix, rtol=0, atol=0)
    assert all(
        torch.equal(model_input[0], expected_prefix) for model_input in seen_inputs
    )
    assert all(seen_requires_grad)

    target = torch.ones((2, 2), dtype=torch.float32)
    unguided_error = torch.linalg.vector_norm(unguided[1:] - target)
    guided_error = torch.linalg.vector_norm(guided[1:] - target)
    assert guided_error < unguided_error
    # Exponential weights make the earlier overlap position move farther.
    assert guided[1, 0] > guided[2, 0] > 0
    assert diagnostics.guidance_mode == "native_denoise_vjp"
    assert diagnostics.rtc_native_guidance is True
    assert diagnostics.rtc_host_postprocess_fallback is False
    assert diagnostics.vjp_steps == 2
    assert len(diagnostics.guidance_losses) == 2


def test_torch_rtc_vjp_contains_the_denoiser_jacobian() -> None:
    torch = pytest.importorskip("torch")
    from rlinf.projects.fibocom_vla.inference.rtc_torch import (
        guided_flow_denoise,
    )

    conditioning = RTCConditioning(
        hard_prefix=np.empty((0, 1), dtype=np.float32),
        overlap_target=np.ones((1, 1), dtype=np.float32),
        overlap_weights=np.ones(1, dtype=np.float32),
    )

    def velocity(actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return 0.5 * actions

    guided, diagnostics = guided_flow_denoise(
        velocity,
        torch.zeros((1, 1), dtype=torch.float32),
        [1.0, 0.0],
        conditioning,
        guidance_scale=1.0,
    )

    # At t=1, x0_hat=x-0.5*x. Therefore dL/dx=(x0_hat-1)*0.5=-0.5
    # at x=0, and dt=-1 moves the sample to +0.5. Treating the denoiser as
    # stop-gradient would incorrectly move it to +1.0.
    torch.testing.assert_close(guided, torch.tensor([[0.5]]))
    assert diagnostics.vjp_steps == 1
