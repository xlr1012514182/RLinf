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

"""Optional Triton core for continuous speculative verification.

The main VLA still produces the parallel velocity tensor.  The optional Triton
kernel accelerates only interpolation, clean-action reconstruction, and
per-step absolute/relative RMS.  Compliance thresholds, contiguous-prefix
reduction, verifier averaging, gripper guards, and draft/main stitching remain
in the Torch ``_finalize`` path.  The full Torch implementation is the numerical
reference.  Triton is imported only when a CUDA tensor requests it, and every
unavailable or failed acceleration attempt reports an explicit fallback reason.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..errors import ShapeMismatchError

Backend = Literal["auto", "torch", "triton"]


@dataclass(frozen=True)
class TritonSpeculativeConfig:
    """Thresholds and safety gates for one continuous-action verification."""

    absolute_error_threshold: float
    relative_error_threshold: float
    distance_indices: tuple[int, ...] | None = None
    evaluation_horizon: int | None = None
    relative_floor: float = 1e-6
    gripper_index: int | None = None
    gripper_switch_threshold: float | None = None
    exclude_gripper_from_radius: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("absolute_error_threshold", self.absolute_error_threshold),
            ("relative_error_threshold", self.relative_error_threshold),
        ):
            if not torch.isfinite(torch.tensor(value)).item() or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not torch.isfinite(torch.tensor(self.relative_floor)).item()
            or self.relative_floor <= 0
        ):
            raise ValueError("relative_floor must be finite and positive")
        if self.evaluation_horizon is not None and self.evaluation_horizon <= 0:
            raise ValueError("evaluation_horizon must be positive when provided")
        gripper_fields = (
            self.gripper_index is not None,
            self.gripper_switch_threshold is not None,
        )
        if any(gripper_fields) and not all(gripper_fields):
            raise ValueError(
                "gripper_index and gripper_switch_threshold must be set together"
            )
        if self.gripper_index is not None and self.gripper_index < 0:
            raise ValueError("gripper_index must be non-negative")
        if (
            self.gripper_switch_threshold is not None
            and not torch.isfinite(torch.tensor(self.gripper_switch_threshold)).item()
        ):
            raise ValueError("gripper_switch_threshold must be finite")
        if self.distance_indices is not None:
            if not self.distance_indices:
                raise ValueError("distance_indices must not be empty")
            if any(index < 0 for index in self.distance_indices):
                raise ValueError("distance_indices must be non-negative")
            if len(set(self.distance_indices)) != len(self.distance_indices):
                raise ValueError("distance_indices must be unique")


@dataclass(frozen=True)
class TritonSpeculativeResult:
    """Auditable result of one fused speculative post-processing pass."""

    interpolation: torch.Tensor
    x0_hat: torch.Tensor
    absolute_error: torch.Tensor
    relative_error: torch.Tensor
    compliant: torch.Tensor
    prefix_lengths: torch.Tensor
    radius_prefix: int
    accepted_prefix: int
    main_candidate: torch.Tensor
    stitched_actions: torch.Tensor
    distance_indices: tuple[int, ...]
    evaluated_horizon: int
    backend: Literal["torch", "triton"]
    requested_backend: Backend
    fallback_reason: str | None
    gripper_guard_active: bool
    gripper_guard_reason: str
    gripper_crossing_ks: tuple[int, ...] = ()
    gripper_first_crossing_step: int | None = None
    gripper_post_stitch_cut: int | None = None

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return stable, serialization-friendly execution diagnostics."""

        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "fallback_reason": self.fallback_reason,
            "verification_count": int(self.x0_hat.shape[0]),
            "action_horizon": int(self.x0_hat.shape[1]),
            "action_dim": int(self.x0_hat.shape[2]),
            "distance_indices": self.distance_indices,
            "evaluated_horizon": self.evaluated_horizon,
            "prefix_lengths_per_verify": tuple(
                int(value) for value in self.prefix_lengths.detach().cpu().tolist()
            ),
            "radius_prefix": self.radius_prefix,
            "accepted_prefix": self.accepted_prefix,
            "acceptance_reduction": "contiguous_per_k_then_min",
            "tail_reduction": "mean_k_x0_hat",
            "gripper_guard_active": self.gripper_guard_active,
            "gripper_guard_reason": self.gripper_guard_reason,
            "gripper_crossing_ks": self.gripper_crossing_ks,
            "gripper_first_crossing_step": self.gripper_first_crossing_step,
            "gripper_post_stitch_cut": self.gripper_post_stitch_cut,
        }


@dataclass(frozen=True)
class _PreparedInputs:
    draft: torch.Tensor
    noise: torch.Tensor
    velocity: torch.Tensor
    times: torch.Tensor
    distance_indices: tuple[int, ...]
    distance_mask: torch.Tensor
    evaluated_horizon: int
    previous_gripper_value: float | None


def _ensure_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(
            f"{name} contains non-finite values; speculative output rejected"
        )


def _prepare_inputs(
    draft: torch.Tensor,
    noise: torch.Tensor,
    predicted_velocity: torch.Tensor,
    verification_times: torch.Tensor,
    config: TritonSpeculativeConfig,
    previous_gripper_value: float | None,
) -> _PreparedInputs:
    if not isinstance(draft, torch.Tensor) or draft.ndim != 2:
        raise ShapeMismatchError("draft must be a Torch tensor shaped [H, A]")
    if draft.shape[0] <= 0 or draft.shape[1] <= 0:
        raise ShapeMismatchError("draft horizon and action dimension must be positive")
    if not isinstance(noise, torch.Tensor) or noise.shape != draft.shape:
        raise ShapeMismatchError("noise must be a Torch tensor matching draft [H, A]")
    if not isinstance(predicted_velocity, torch.Tensor) or predicted_velocity.ndim != 3:
        raise ShapeMismatchError(
            "predicted_velocity must be a Torch tensor shaped [K, H, A]"
        )
    if tuple(predicted_velocity.shape[1:]) != tuple(draft.shape):
        raise ShapeMismatchError("predicted_velocity [H, A] must match draft")
    if predicted_velocity.shape[0] <= 0:
        raise ShapeMismatchError("predicted_velocity must contain at least one K")
    if not isinstance(verification_times, torch.Tensor):
        raise ShapeMismatchError("verification_times must be a Torch tensor")
    if (
        verification_times.ndim != 1
        or verification_times.shape[0] != (predicted_velocity.shape[0])
    ):
        raise ShapeMismatchError("verification_times must be [K]")

    device = draft.device
    work_draft = draft.to(device=device, dtype=torch.float32).contiguous()
    work_noise = noise.to(device=device, dtype=torch.float32).contiguous()
    work_velocity = predicted_velocity.to(
        device=device, dtype=torch.float32
    ).contiguous()
    work_times = verification_times.to(device=device, dtype=torch.float32).contiguous()
    _ensure_finite(work_draft, "draft")
    _ensure_finite(work_noise, "noise")
    _ensure_finite(work_velocity, "predicted_velocity")
    _ensure_finite(work_times, "verification_times")
    if bool(((work_times <= 0) | (work_times > 1)).any().item()):
        raise ValueError("verification_times must lie in (0, 1]")

    horizon, action_dim = (int(value) for value in draft.shape)
    evaluated_horizon = config.evaluation_horizon or horizon
    if evaluated_horizon > horizon:
        raise ShapeMismatchError("evaluation_horizon exceeds the draft horizon")
    if config.gripper_index is not None and config.gripper_index >= action_dim:
        raise ShapeMismatchError("gripper_index exceeds the action dimension")

    if previous_gripper_value is not None:
        previous_gripper_value = float(previous_gripper_value)
        if not torch.isfinite(torch.tensor(previous_gripper_value)).item():
            raise ValueError(
                "previous_gripper_value is non-finite; speculative output rejected"
            )
    gripper_active = (
        config.gripper_index is not None and previous_gripper_value is not None
    )

    indices = (
        tuple(range(action_dim))
        if config.distance_indices is None
        else config.distance_indices
    )
    if any(index >= action_dim for index in indices):
        raise ShapeMismatchError("distance_indices exceed the action dimension")
    if (
        gripper_active
        and config.exclude_gripper_from_radius
        and config.gripper_index is not None
    ):
        indices = tuple(index for index in indices if index != config.gripper_index)
    if not indices:
        raise ShapeMismatchError("no action dimensions remain for RMS verification")
    distance_mask = torch.zeros(action_dim, device=device, dtype=torch.int8)
    distance_mask[list(indices)] = 1
    return _PreparedInputs(
        draft=work_draft,
        noise=work_noise,
        velocity=work_velocity,
        times=work_times,
        distance_indices=indices,
        distance_mask=distance_mask.contiguous(),
        evaluated_horizon=evaluated_horizon,
        previous_gripper_value=previous_gripper_value,
    )


def _torch_core(
    prepared: _PreparedInputs, config: TritonSpeculativeConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    times = prepared.times[:, None, None]
    interpolation = times * prepared.noise[None] + (1.0 - times) * prepared.draft[None]
    x0_hat = interpolation - times * prepared.velocity
    indices = torch.tensor(
        prepared.distance_indices, device=prepared.draft.device, dtype=torch.long
    )
    difference = (
        x0_hat.index_select(2, indices) - prepared.draft.index_select(1, indices)[None]
    )
    absolute_error = torch.sqrt(torch.mean(difference.square(), dim=-1))
    draft_scale = torch.sqrt(
        torch.mean(
            prepared.draft.index_select(1, indices).square(),
            dim=-1,
        )
    ).clamp_min(config.relative_floor)
    relative_error = absolute_error / draft_scale[None]
    return interpolation, x0_hat, absolute_error, relative_error


# These globals are populated only after a CUDA caller asks for Triton.  The
# undecorated kernel keeps importing this module safe on CPU-only hosts.
tl: Any | None = None
_triton_kernel: Any | None = None


def _speculative_rms_kernel(
    draft_pointer,
    noise_pointer,
    velocity_pointer,
    times_pointer,
    distance_mask_pointer,
    interpolation_pointer,
    x0_pointer,
    absolute_pointer,
    relative_pointer,
    horizon,
    action_dim,
    distance_dim,
    relative_floor,
    BLOCK_A,
):
    """Triton program body; decorated lazily by :func:`_load_triton_kernel`."""

    program = tl.program_id(0)
    verify_index = program // horizon
    horizon_index = program - verify_index * horizon
    action_offsets = tl.arange(0, BLOCK_A)
    action_mask = action_offsets < action_dim
    draft_offsets = horizon_index * action_dim + action_offsets
    velocity_offsets = program * action_dim + action_offsets
    draft_values = tl.load(draft_pointer + draft_offsets, mask=action_mask, other=0.0)
    noise_values = tl.load(noise_pointer + draft_offsets, mask=action_mask, other=0.0)
    velocity_values = tl.load(
        velocity_pointer + velocity_offsets, mask=action_mask, other=0.0
    )
    verification_time = tl.load(times_pointer + verify_index)
    interpolation = (
        verification_time * noise_values + (1.0 - verification_time) * draft_values
    )
    x0_hat = interpolation - verification_time * velocity_values
    tl.store(interpolation_pointer + velocity_offsets, interpolation, mask=action_mask)
    tl.store(x0_pointer + velocity_offsets, x0_hat, mask=action_mask)

    distance_mask = (
        tl.load(distance_mask_pointer + action_offsets, mask=action_mask, other=0) != 0
    )
    difference = tl.where(distance_mask, x0_hat - draft_values, 0.0)
    draft_for_radius = tl.where(distance_mask, draft_values, 0.0)
    inverse_dims = 1.0 / distance_dim
    absolute_error = tl.sqrt(tl.sum(difference * difference, axis=0) * inverse_dims)
    draft_scale = tl.sqrt(
        tl.sum(draft_for_radius * draft_for_radius, axis=0) * inverse_dims
    )
    draft_scale = tl.maximum(draft_scale, relative_floor)
    tl.store(absolute_pointer + program, absolute_error)
    tl.store(relative_pointer + program, absolute_error / draft_scale)


def _load_triton_kernel() -> tuple[Any | None, str | None]:
    global _triton_kernel, tl
    if _triton_kernel is not None:
        return _triton_kernel, None
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except (ImportError, ModuleNotFoundError):
        return None, "triton_unavailable"
    try:
        # BLOCK_A drives ``tl.arange`` and therefore must be a compile-time
        # meta-parameter.  Assigning the marker only after the lazy import
        # keeps module import safe when Triton is not installed.
        _speculative_rms_kernel.__annotations__["BLOCK_A"] = tl.constexpr
        _triton_kernel = triton.jit(_speculative_rms_kernel)
    except Exception:
        tl = None
        return None, "triton_kernel_initialization_failed"
    return _triton_kernel, None


def _triton_core(
    prepared: _PreparedInputs, config: TritonSpeculativeConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel, unavailable_reason = _load_triton_kernel()
    if kernel is None:
        raise RuntimeError(unavailable_reason or "triton_unavailable")
    verification_count, horizon, action_dim = prepared.velocity.shape
    interpolation = torch.empty_like(prepared.velocity)
    x0_hat = torch.empty_like(prepared.velocity)
    absolute_error = torch.empty(
        (verification_count, horizon),
        device=prepared.draft.device,
        dtype=torch.float32,
    )
    relative_error = torch.empty_like(absolute_error)
    block_action_dim = 1 << (int(action_dim) - 1).bit_length()
    kernel[(int(verification_count * horizon),)](
        prepared.draft,
        prepared.noise,
        prepared.velocity,
        prepared.times,
        prepared.distance_mask,
        interpolation,
        x0_hat,
        absolute_error,
        relative_error,
        int(horizon),
        int(action_dim),
        len(prepared.distance_indices),
        float(config.relative_floor),
        BLOCK_A=block_action_dim,
    )
    return interpolation, x0_hat, absolute_error, relative_error


def _crossings(
    values: torch.Tensor, previous_value: float, threshold: float
) -> torch.Tensor:
    leading = torch.full(
        (*values.shape[:-1], 1),
        previous_value,
        device=values.device,
        dtype=values.dtype,
    )
    previous = torch.cat((leading, values[..., :-1]), dim=-1)
    return ((previous < threshold) & (values >= threshold)) | (
        (previous >= threshold) & (values < threshold)
    )


def _finalize(
    prepared: _PreparedInputs,
    config: TritonSpeculativeConfig,
    core: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    backend: Literal["torch", "triton"],
    requested_backend: Backend,
    fallback_reason: str | None,
) -> TritonSpeculativeResult:
    interpolation, x0_hat, absolute_error, relative_error = core
    for name, tensor in (
        ("interpolation", interpolation),
        ("x0_hat", x0_hat),
        ("absolute_error", absolute_error),
        ("relative_error", relative_error),
    ):
        _ensure_finite(tensor, name)

    compliant = (absolute_error <= config.absolute_error_threshold) | (
        relative_error <= config.relative_error_threshold
    )
    if prepared.evaluated_horizon < prepared.draft.shape[0]:
        compliant = compliant.clone()
        compliant[:, prepared.evaluated_horizon :] = False
        absolute_error = absolute_error.clone()
        relative_error = relative_error.clone()
        absolute_error[:, prepared.evaluated_horizon :] = torch.inf
        relative_error[:, prepared.evaluated_horizon :] = torch.inf
    contiguous = torch.cumprod(compliant.to(dtype=torch.int64), dim=1)
    prefix_lengths = contiguous.sum(dim=1, dtype=torch.int64)
    radius_prefix = int(prefix_lengths.min().item())

    main_candidate = x0_hat.mean(dim=0)
    accepted_prefix = radius_prefix
    gripper_active = (
        config.gripper_index is not None and prepared.previous_gripper_value is not None
    )
    gripper_reason = "not_configured"
    crossing_ks: tuple[int, ...] = ()
    first_crossing_step: int | None = None
    post_stitch_cut: int | None = None
    if config.gripper_index is not None and prepared.previous_gripper_value is None:
        gripper_reason = "previous_value_unavailable"
    elif gripper_active:
        assert config.gripper_index is not None
        assert config.gripper_switch_threshold is not None
        assert prepared.previous_gripper_value is not None
        values = x0_hat[:, : prepared.evaluated_horizon, config.gripper_index]
        verifier_crossings = _crossings(
            values,
            prepared.previous_gripper_value,
            config.gripper_switch_threshold,
        )
        crossing_tensor = verifier_crossings.any(dim=1).nonzero().flatten()
        crossing_ks = tuple(
            int(value) for value in crossing_tensor.detach().cpu().tolist()
        )
        if crossing_ks:
            first_crossing_step = int(verifier_crossings.nonzero()[:, 1].min().item())
            accepted_prefix = 0
            gripper_reason = "verifier_crossing"
        else:
            gripper_reason = "enabled_no_crossing"

    stitched = main_candidate.clone()
    stitched[:accepted_prefix] = prepared.draft[:accepted_prefix]
    if gripper_active and not crossing_ks and accepted_prefix > 0:
        assert config.gripper_index is not None
        assert config.gripper_switch_threshold is not None
        assert prepared.previous_gripper_value is not None
        stitched_crossings = _crossings(
            stitched[:accepted_prefix, config.gripper_index].unsqueeze(0),
            prepared.previous_gripper_value,
            config.gripper_switch_threshold,
        )[0]
        crossing_steps = stitched_crossings.nonzero().flatten()
        if crossing_steps.numel():
            post_stitch_cut = int(crossing_steps[0].item())
            accepted_prefix = post_stitch_cut
            stitched = main_candidate.clone()
            stitched[:accepted_prefix] = prepared.draft[:accepted_prefix]
            gripper_reason = "stitched_prefix_crossing"

    return TritonSpeculativeResult(
        interpolation=interpolation,
        x0_hat=x0_hat,
        absolute_error=absolute_error,
        relative_error=relative_error,
        compliant=compliant,
        prefix_lengths=prefix_lengths,
        radius_prefix=radius_prefix,
        accepted_prefix=accepted_prefix,
        main_candidate=main_candidate,
        stitched_actions=stitched,
        distance_indices=prepared.distance_indices,
        evaluated_horizon=prepared.evaluated_horizon,
        backend=backend,
        requested_backend=requested_backend,
        fallback_reason=fallback_reason,
        gripper_guard_active=gripper_active,
        gripper_guard_reason=gripper_reason,
        gripper_crossing_ks=crossing_ks,
        gripper_first_crossing_step=first_crossing_step,
        gripper_post_stitch_cut=post_stitch_cut,
    )


def torch_speculative_reference(
    draft: torch.Tensor,
    noise: torch.Tensor,
    predicted_velocity: torch.Tensor,
    verification_times: torch.Tensor,
    config: TritonSpeculativeConfig,
    *,
    previous_gripper_value: float | None = None,
) -> TritonSpeculativeResult:
    """Run the authoritative Torch implementation."""

    prepared = _prepare_inputs(
        draft,
        noise,
        predicted_velocity,
        verification_times,
        config,
        previous_gripper_value,
    )
    return _finalize(
        prepared,
        config,
        _torch_core(prepared, config),
        backend="torch",
        requested_backend="torch",
        fallback_reason=None,
    )


def speculative_verify_and_stitch(
    draft: torch.Tensor,
    noise: torch.Tensor,
    predicted_velocity: torch.Tensor,
    verification_times: torch.Tensor,
    config: TritonSpeculativeConfig,
    *,
    previous_gripper_value: float | None = None,
    backend: Backend = "auto",
) -> TritonSpeculativeResult:
    """Run Triton when possible and otherwise report an explicit Torch fallback."""

    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    prepared = _prepare_inputs(
        draft,
        noise,
        predicted_velocity,
        verification_times,
        config,
        previous_gripper_value,
    )
    if backend == "torch":
        return _finalize(
            prepared,
            config,
            _torch_core(prepared, config),
            backend="torch",
            requested_backend=backend,
            fallback_reason=None,
        )

    fallback_reason: str | None = None
    if not prepared.draft.is_cuda:
        fallback_reason = "cuda_input_required"
    else:
        kernel, unavailable_reason = _load_triton_kernel()
        if kernel is None:
            fallback_reason = unavailable_reason or "triton_unavailable"
        else:
            try:
                return _finalize(
                    prepared,
                    config,
                    _triton_core(prepared, config),
                    backend="triton",
                    requested_backend=backend,
                    fallback_reason=None,
                )
            except Exception as error:
                fallback_reason = f"triton_execution_failed:{type(error).__name__}"

    return _finalize(
        prepared,
        config,
        _torch_core(prepared, config),
        backend="torch",
        requested_backend=backend,
        fallback_reason=fallback_reason,
    )


__all__ = [
    "TritonSpeculativeConfig",
    "TritonSpeculativeResult",
    "speculative_verify_and_stitch",
    "torch_speculative_reference",
]
