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

"""Numerical and fail-closed tests for speculative Triton post-processing."""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from rlinf.projects.fibocom_vla.inference.triton_speculative import (  # noqa: E402
    TritonSpeculativeConfig,
    speculative_verify_and_stitch,
    torch_speculative_reference,
)


def _velocity_for_x0(
    draft: torch.Tensor,
    noise: torch.Tensor,
    times: torch.Tensor,
    desired_x0: torch.Tensor,
) -> torch.Tensor:
    interpolation = (
        times[:, None, None] * noise[None] + (1.0 - times[:, None, None]) * draft[None]
    )
    return (interpolation - desired_x0) / times[:, None, None]


def _h50_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260825)
    draft = torch.randn((50, 7), generator=generator)
    noise = torch.randn((50, 7), generator=generator)
    times = torch.tensor([0.10, 0.05], dtype=torch.float32)
    return draft, noise, times


def _config(**overrides: object) -> TritonSpeculativeConfig:
    values: dict[str, object] = {
        "absolute_error_threshold": 0.05,
        "relative_error_threshold": 0.05,
    }
    values.update(overrides)
    return TritonSpeculativeConfig(**values)


def test_h50_a7_k2_cpu_fallback_matches_torch_reference() -> None:
    draft, noise, times = _h50_inputs()
    offsets = torch.linspace(0.0, 0.08, 50).view(1, 50, 1)
    desired_x0 = draft[None] + offsets.expand(2, 50, 7)
    velocity = _velocity_for_x0(draft, noise, times, desired_x0)
    config = _config()

    reference = torch_speculative_reference(draft, noise, velocity, times, config)
    result = speculative_verify_and_stitch(
        draft, noise, velocity, times, config, backend="triton"
    )

    assert result.x0_hat.shape == (2, 50, 7)
    assert result.absolute_error.shape == (2, 50)
    assert result.backend == "torch"
    assert result.requested_backend == "triton"
    assert result.fallback_reason == "cuda_input_required"
    torch.testing.assert_close(result.interpolation, reference.interpolation)
    torch.testing.assert_close(result.x0_hat, reference.x0_hat)
    torch.testing.assert_close(result.absolute_error, reference.absolute_error)
    torch.testing.assert_close(result.relative_error, reference.relative_error)
    torch.testing.assert_close(result.stitched_actions, reference.stitched_actions)


@pytest.mark.parametrize(
    ("error_pattern", "expected_prefix"),
    [
        ("zero", 0),
        ("full", 50),
        ("non_contiguous", 2),
    ],
)
def test_contiguous_prefix_reduction(error_pattern: str, expected_prefix: int) -> None:
    draft, noise, times = _h50_inputs()
    desired_x0 = draft[None].expand(2, -1, -1).clone()
    if error_pattern == "zero":
        desired_x0 += 1.0
    elif error_pattern == "non_contiguous":
        # K=0 rejects step 2 but later points comply. K=1 rejects only at
        # step 4. The contiguous K prefixes are therefore (2, 4), not counts
        # of all compliant points, and min(K) must be two.
        desired_x0[0, 2] += 1.0
        desired_x0[1, 4] += 1.0
    velocity = _velocity_for_x0(draft, noise, times, desired_x0)

    result = torch_speculative_reference(
        draft,
        noise,
        velocity,
        times,
        _config(absolute_error_threshold=0.1, relative_error_threshold=0.1),
    )

    assert result.accepted_prefix == expected_prefix
    assert result.radius_prefix == expected_prefix
    if error_pattern == "zero":
        assert tuple(result.prefix_lengths.tolist()) == (0, 0)
        torch.testing.assert_close(result.stitched_actions, result.main_candidate)
    elif error_pattern == "full":
        assert tuple(result.prefix_lengths.tolist()) == (50, 50)
        torch.testing.assert_close(result.stitched_actions, draft)
    else:
        assert tuple(result.prefix_lengths.tolist()) == (2, 4)
        assert bool(result.compliant[0, 10].item())
        torch.testing.assert_close(result.stitched_actions[:2], draft[:2])
        torch.testing.assert_close(
            result.stitched_actions[2:], result.main_candidate[2:]
        )


def test_rms_uses_selected_dimensions_and_excludes_live_gripper() -> None:
    draft = torch.ones((3, 7), dtype=torch.float32)
    noise = torch.zeros_like(draft)
    times = torch.tensor([0.10, 0.05])
    desired_x0 = draft[None].expand(2, -1, -1).clone()
    desired_x0[:, :, 0] += 2.0
    desired_x0[:, :, 6] += 100.0
    velocity = _velocity_for_x0(draft, noise, times, desired_x0)

    result = torch_speculative_reference(
        draft,
        noise,
        velocity,
        times,
        _config(
            absolute_error_threshold=1.0,
            relative_error_threshold=1.0,
            gripper_index=6,
            gripper_switch_threshold=200.0,
        ),
        previous_gripper_value=1.0,
    )

    assert result.distance_indices == (0, 1, 2, 3, 4, 5)
    expected_rms = torch.full((2, 3), 2.0 / (6.0**0.5))
    torch.testing.assert_close(result.absolute_error, expected_rms)


def test_any_k_gripper_crossing_rejects_the_draft_prefix() -> None:
    draft = torch.zeros((50, 7), dtype=torch.float32)
    draft[:, 6] = 0.1
    noise = torch.zeros_like(draft)
    times = torch.tensor([0.10, 0.05])
    desired_x0 = draft[None].expand(2, -1, -1).clone()
    desired_x0[1, 20:, 6] = 0.8
    velocity = _velocity_for_x0(draft, noise, times, desired_x0)

    result = torch_speculative_reference(
        draft,
        noise,
        velocity,
        times,
        _config(gripper_index=6, gripper_switch_threshold=0.5),
        previous_gripper_value=0.1,
    )

    assert result.radius_prefix == 50
    assert result.accepted_prefix == 0
    assert result.gripper_guard_active is True
    assert result.gripper_guard_reason == "verifier_crossing"
    assert result.gripper_crossing_ks == (1,)
    assert result.gripper_first_crossing_step == 20
    torch.testing.assert_close(result.stitched_actions, result.main_candidate)


def test_gripper_crossing_in_accepted_draft_restitches_tail() -> None:
    draft = torch.zeros((6, 7), dtype=torch.float32)
    draft[:, 6] = 0.49
    draft[3:, 6] = 0.51
    noise = torch.zeros_like(draft)
    times = torch.tensor([0.10, 0.05])
    # Both verifier candidates remain below the switch threshold. The radius
    # accepts the full draft, but the draft itself crosses at action three.
    desired_x0 = draft[None].expand(2, -1, -1).clone()
    desired_x0[:, :, 6] = 0.49
    velocity = _velocity_for_x0(draft, noise, times, desired_x0)

    result = torch_speculative_reference(
        draft,
        noise,
        velocity,
        times,
        _config(gripper_index=6, gripper_switch_threshold=0.5),
        previous_gripper_value=0.49,
    )

    assert result.radius_prefix == 6
    assert result.gripper_crossing_ks == ()
    assert result.gripper_post_stitch_cut == 3
    assert result.accepted_prefix == 3
    assert result.gripper_guard_reason == "stitched_prefix_crossing"
    torch.testing.assert_close(result.stitched_actions[:3], draft[:3])
    torch.testing.assert_close(result.stitched_actions[3:], result.main_candidate[3:])


@pytest.mark.parametrize("input_name", ["draft", "noise", "velocity", "times"])
def test_non_finite_input_fails_closed_without_actions(input_name: str) -> None:
    draft, noise, times = _h50_inputs()
    desired_x0 = draft[None].expand(2, -1, -1).clone()
    velocity = _velocity_for_x0(draft, noise, times, desired_x0)
    tensors = {
        "draft": draft,
        "noise": noise,
        "velocity": velocity,
        "times": times,
    }
    tensors[input_name] = tensors[input_name].clone()
    tensors[input_name].view(-1)[0] = torch.nan

    with pytest.raises(ValueError, match="speculative output rejected"):
        speculative_verify_and_stitch(
            tensors["draft"],
            tensors["noise"],
            tensors["velocity"],
            tensors["times"],
            _config(),
            backend="auto",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required for kernel parity",
)
def test_cuda_triton_kernel_matches_reference() -> None:
    draft, noise, times = (value.cuda() for value in _h50_inputs())
    desired_x0 = draft[None].expand(2, -1, -1).clone()
    desired_x0[:, 30:] += 0.2
    velocity = _velocity_for_x0(draft, noise, times, desired_x0)
    config = _config(absolute_error_threshold=0.1, relative_error_threshold=0.1)

    reference = torch_speculative_reference(draft, noise, velocity, times, config)
    result = speculative_verify_and_stitch(
        draft, noise, velocity, times, config, backend="triton"
    )

    assert result.backend == "triton"
    assert result.fallback_reason is None
    torch.testing.assert_close(result.interpolation, reference.interpolation)
    torch.testing.assert_close(result.x0_hat, reference.x0_hat)
    torch.testing.assert_close(
        result.absolute_error, reference.absolute_error, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        result.relative_error, reference.relative_error, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(result.stitched_actions, reference.stitched_actions)
