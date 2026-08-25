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

"""Differentiable RTC guidance executed inside a PyTorch flow denoiser.

Unlike :func:`rtc.apply_rtc_conditioning`, this module does not blend a
finished action chunk on the host.  It freezes the committed prefix before and
after every Euler step and computes a vector-Jacobian product (VJP) through the
denoiser's clean-action estimate for the still-overlapping old-chunk target.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..errors import ShapeMismatchError
from .rtc import RTCConditioning

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RTCDenoiseDiagnostics:
    """Evidence that distinguishes native guidance from host postprocessing."""

    guidance_mode: str
    denoise_steps: int
    vjp_steps: int
    hard_prefix_length: int
    overlap_length: int
    guidance_scale: float
    guidance_losses: tuple[float, ...]
    timestep_schedule: tuple[float, ...]
    rtc_native_guidance: bool = True
    rtc_host_postprocess_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics for ``PolicyOutput``."""

        return asdict(self)


def exponential_position_weights(
    length: int,
    *,
    initial_weight: float,
    decay: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create ``initial_weight * decay**position`` overlap weights."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if initial_weight < 0:
        raise ValueError("initial_weight must be non-negative")
    if not 0 < decay <= 1:
        raise ValueError("decay must lie in (0, 1]")
    positions = torch.arange(length, device=device, dtype=dtype)
    base = torch.as_tensor(decay, device=device, dtype=dtype)
    return torch.as_tensor(initial_weight, device=device, dtype=dtype) * torch.pow(
        base, positions
    )


def _as_batched(actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if actions.ndim == 2:
        return actions.unsqueeze(0), True
    if actions.ndim == 3:
        return actions, False
    raise ShapeMismatchError("flow actions must be [T, A] or [B, T, A]")


def _freeze_prefix(actions: torch.Tensor, hard_prefix: torch.Tensor) -> torch.Tensor:
    hard_length = int(hard_prefix.shape[0])
    if hard_length == 0:
        return actions
    batch_prefix = hard_prefix.unsqueeze(0).expand(actions.shape[0], -1, -1)
    return torch.cat((batch_prefix, actions[:, hard_length:, :]), dim=1)


def _call_velocity(
    velocity_fn: VelocityFn,
    actions: torch.Tensor,
    timestep: torch.Tensor,
    *,
    return_unbatched: bool,
) -> torch.Tensor:
    model_input = actions[0] if return_unbatched else actions
    velocity = velocity_fn(model_input, timestep)
    if not isinstance(velocity, torch.Tensor):
        raise TypeError("velocity_fn must return a torch.Tensor")
    if return_unbatched:
        if velocity.ndim != 2:
            raise ShapeMismatchError("unbatched velocity must be [T, A]")
        velocity = velocity.unsqueeze(0)
    if velocity.shape != actions.shape:
        raise ShapeMismatchError(
            f"velocity shape {tuple(velocity.shape)} does not match "
            f"flow state {tuple(actions.shape)}"
        )
    return velocity


def guided_flow_denoise(
    velocity_fn: VelocityFn,
    initial_noise: torch.Tensor,
    timesteps: torch.Tensor | Sequence[float],
    conditioning: RTCConditioning,
    *,
    guidance_scale: float = 1.0,
    create_graph: bool = False,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, RTCDenoiseDiagnostics]:
    """Run Euler flow denoising with hard-prefix freeze and overlap VJP.

    The timestep schedule must be strictly decreasing, normally from 1 to 0.
    At step ``t`` the clean-action estimate follows the same convention as the
    speculative verifier::

        x0_hat = x_t - t * v_theta(x_t, t)

    The weighted overlap objective is::

        L = sum_(b,i) w_i ||x0_hat[b,i] - old_chunk[i]||^2
            / (2 * B * A * sum_i w_i)

    where committed-prefix positions are excluded because they are enforced
    exactly. ``torch.autograd.grad(L, x_t)`` evaluates the denoiser VJP. Since
    Euler integrates from high to low time (``dt < 0``), adding the gradient to
    velocity makes ``dt * guidance`` a gradient-descent update on L.

    ``velocity_fn`` receives the same rank as ``initial_noise``. It should be a
    closure over observation/prompt/KV conditioning. With ``create_graph=False``
    each step detaches after its first-order VJP, which is the inference mode;
    ``create_graph=True`` retains higher-order autograd state for research.
    """

    if not isinstance(initial_noise, torch.Tensor):
        raise TypeError("initial_noise must be a torch.Tensor")
    if not torch.is_floating_point(initial_noise):
        raise TypeError("initial_noise must use a floating dtype")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")

    actions, return_unbatched = _as_batched(initial_noise)
    batch_size, horizon, action_dim = (int(value) for value in actions.shape)
    schedule = torch.as_tensor(timesteps, device=actions.device, dtype=actions.dtype)
    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("timesteps must contain at least two scalar values")
    if not bool(torch.isfinite(schedule).all().item()):
        raise ValueError("timesteps must be finite")
    if not bool(((schedule >= 0) & (schedule <= 1)).all().item()):
        raise ValueError("timesteps must lie in [0, 1]")
    if not bool((schedule[:-1] > schedule[1:]).all().item()):
        raise ValueError("timesteps must be strictly decreasing")

    hard_prefix = torch.as_tensor(
        conditioning.hard_prefix,
        device=actions.device,
        dtype=actions.dtype,
    )
    overlap_target = torch.as_tensor(
        conditioning.overlap_target,
        device=actions.device,
        dtype=actions.dtype,
    )
    overlap_weights = torch.as_tensor(
        conditioning.overlap_weights,
        device=actions.device,
        dtype=actions.dtype,
    )
    if hard_prefix.ndim != 2 or overlap_target.ndim != 2:
        raise ShapeMismatchError("RTC targets must be rank-2 [T, A]")
    if hard_prefix.shape[1] != action_dim or overlap_target.shape[1] != action_dim:
        raise ShapeMismatchError("RTC targets and flow state use different action dims")
    hard_length = int(hard_prefix.shape[0])
    overlap_length = min(int(overlap_target.shape[0]), horizon)
    if hard_length > horizon:
        raise ShapeMismatchError("hard RTC prefix exceeds denoising horizon")
    if overlap_weights.shape != (int(overlap_target.shape[0]),):
        raise ShapeMismatchError("RTC overlap weights must match overlap target")

    active_weights = overlap_weights[:overlap_length].clone()
    if hard_length:
        active_weights[: min(hard_length, overlap_length)] = 0
    has_guidance = bool(
        guidance_scale > 0
        and overlap_length > hard_length
        and torch.any(active_weights > 0).item()
    )

    x_t = _freeze_prefix(actions, hard_prefix)
    losses: list[float] = []
    vjp_steps = 0
    for step in range(int(schedule.numel()) - 1):
        timestep = schedule[step]
        dt = schedule[step + 1] - timestep
        x_t = _freeze_prefix(x_t, hard_prefix)

        if has_guidance:
            # enable_grad deliberately overrides an enclosing inference
            # no_grad context: this first-order VJP is the native RTC update.
            with torch.enable_grad():
                current = x_t
                if not current.requires_grad:
                    current = current.detach().requires_grad_(True)
                velocity = _call_velocity(
                    velocity_fn,
                    current,
                    timestep,
                    return_unbatched=return_unbatched,
                )
                clean_estimate = current - timestep * velocity
                error = (
                    clean_estimate[:, :overlap_length, :]
                    - overlap_target[None, :overlap_length, :]
                )
                weighted_error = (
                    error.square().sum(dim=-1) * active_weights[None, :overlap_length]
                )
                denominator = (
                    active_weights.sum() * float(batch_size * action_dim)
                ).clamp_min(torch.finfo(actions.dtype).eps)
                loss = 0.5 * weighted_error.sum() / denominator
                gradient = torch.autograd.grad(
                    loss,
                    current,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                )[0]

                # The objective can couple positions through the denoiser.
                # The resume contract constrains only the old-chunk overlap,
                # so do not push unrelated suffix positions with that VJP.
                position_mask = torch.zeros_like(gradient)
                position_mask[:, hard_length:overlap_length, :] = 1
                gradient = gradient * position_mask
                guided_velocity = velocity + float(guidance_scale) * gradient
                x_next = current + dt * guided_velocity
                if collect_diagnostics:
                    losses.append(float(loss.detach().cpu().item()))
                vjp_steps += 1
                if not create_graph:
                    x_next = x_next.detach()
        else:
            with torch.set_grad_enabled(create_graph):
                velocity = _call_velocity(
                    velocity_fn,
                    x_t,
                    timestep,
                    return_unbatched=return_unbatched,
                )
                x_next = x_t + dt * velocity
                if not create_graph:
                    x_next = x_next.detach()

        # Re-freeze after Euler so no denoiser output or numerical update can
        # move an already committed action.
        x_t = _freeze_prefix(x_next, hard_prefix)

    output = x_t[0] if return_unbatched else x_t
    diagnostics = RTCDenoiseDiagnostics(
        guidance_mode="native_denoise_vjp",
        denoise_steps=int(schedule.numel()) - 1,
        vjp_steps=vjp_steps,
        hard_prefix_length=hard_length,
        overlap_length=overlap_length,
        guidance_scale=float(guidance_scale),
        guidance_losses=tuple(losses),
        timestep_schedule=tuple(
            float(value) for value in schedule.detach().cpu().tolist()
        ),
    )
    return output, diagnostics


__all__ = [
    "RTCDenoiseDiagnostics",
    "exponential_position_weights",
    "guided_flow_denoise",
]
