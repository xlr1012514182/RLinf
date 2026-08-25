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

"""Native RTC sampling adapter for the release/v0.2 RLinf OpenPI model.

The release/v0.2 sampler is decorated with ``torch.no_grad`` and exposes its
denoiser through ``get_suffix_out`` followed by ``action_out_proj``.  This
adapter deliberately enters the denoising loop itself so that overlap guidance
is a real vector-Jacobian product (VJP) through that denoiser.  It does not
post-blend an already completed action chunk.

All previous actions and RTC targets handled here are in normalized OpenPI
*model action space*, before ``output_transform``.  Applying this adapter to
environment-space joint commands would be semantically incorrect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..errors import ShapeMismatchError
from .rtc_torch import exponential_position_weights

OPENPI_RTC_PREFIX_FEATURE_KEY = "rtc_prefix_feature"
OPENPI_RTC_DIAGNOSTICS_KEY = "rtc_diagnostics"
OPENPI_MODEL_ACTIONS_KEY = "openpi_model_actions"

CountSpec = int | Sequence[int] | torch.Tensor


@dataclass(frozen=True)
class OpenPIRTCContext:
    """Previous model-space chunk and controller-time RTC alignment.

    ``executed_steps`` removes actions already consumed by the robot.
    ``hard_prefix_steps`` counts the still-unexecuted actions that are already
    committed and must remain exact while the new chunk is denoised.  The rest
    of the overlap is guided with either explicit ``overlap_weights`` aligned
    to ``previous_model_actions`` or ``guidance_decay**position``. The overall
    VJP multiplier is ``guidance_scale``.
    """

    previous_model_actions: torch.Tensor
    executed_steps: CountSpec = 0
    hard_prefix_steps: CountSpec = 0
    overlap_horizon: int | None = None
    overlap_weights: torch.Tensor | Sequence[float] | None = None
    guidance_scale: float = 1.0
    guidance_decay: float = 0.82

    def __post_init__(self) -> None:
        previous = self.previous_model_actions
        if not isinstance(previous, torch.Tensor):
            raise TypeError("previous_model_actions must be a torch.Tensor")
        if previous.ndim != 3 or previous.shape[0] <= 0 or previous.shape[2] <= 0:
            raise ShapeMismatchError(
                "previous_model_actions must be [B, H_previous, A_model]"
            )
        if not torch.is_floating_point(previous):
            raise TypeError("previous_model_actions must use a floating dtype")
        if not bool(torch.isfinite(previous).all().item()):
            raise ValueError("previous_model_actions contain NaN or infinity")
        if self.overlap_horizon is not None and self.overlap_horizon < 0:
            raise ValueError("overlap_horizon must be non-negative")
        if self.overlap_weights is not None:
            weights = torch.as_tensor(self.overlap_weights).detach()
            if weights.ndim not in (1, 2):
                raise ShapeMismatchError("overlap_weights must be [H] or [B, H]")
            if not torch.is_floating_point(weights):
                weights = weights.to(dtype=torch.float32)
            if not bool(torch.isfinite(weights).all().item()):
                raise ValueError("overlap_weights must be finite")
            if bool((weights < 0).any().item()):
                raise ValueError("overlap_weights must be non-negative")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if not math.isfinite(self.guidance_decay) or not 0 < self.guidance_decay <= 1:
            raise ValueError("guidance_decay must lie in (0, 1]")


@dataclass(frozen=True)
class _PreparedRTC:
    target: torch.Tensor
    hard_mask: torch.Tensor
    soft_weights: torch.Tensor
    executed_steps: tuple[int, ...]
    hard_prefix_lengths: tuple[int, ...]
    overlap_lengths: tuple[int, ...]


def _expand_counts(value: CountSpec, batch_size: int, name: str) -> tuple[int, ...]:
    try:
        counts = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be an integer or length-B integer sequence"
        ) from exc
    if counts.dtype == torch.bool:
        raise TypeError(f"{name} must not use boolean values")
    if counts.ndim == 0:
        counts = counts.expand(batch_size)
    elif counts.ndim != 1 or counts.numel() != batch_size:
        raise ShapeMismatchError(f"{name} must be scalar or have shape [B]")
    if torch.is_floating_point(counts):
        if not bool(torch.isfinite(counts).all().item()):
            raise ValueError(f"{name} must be finite")
        if not bool((counts == counts.round()).all().item()):
            raise ValueError(f"{name} must contain integers")
    result = tuple(int(item) for item in counts.tolist())
    if any(item < 0 for item in result):
        raise ValueError(f"{name} must be non-negative")
    return result


def _prepare_rtc(
    context: OpenPIRTCContext,
    *,
    batch_size: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedRTC:
    previous = context.previous_model_actions
    if int(previous.shape[0]) != batch_size:
        raise ShapeMismatchError("previous RTC batch does not match OpenPI observation")
    if int(previous.shape[2]) != action_dim:
        raise ShapeMismatchError(
            "previous RTC actions and OpenPI model use different action dimensions"
        )
    executed_steps = _expand_counts(
        context.executed_steps, batch_size, "executed_steps"
    )
    requested_hard = _expand_counts(
        context.hard_prefix_steps, batch_size, "hard_prefix_steps"
    )
    previous_horizon = int(previous.shape[1])
    if any(item > previous_horizon for item in executed_steps):
        raise ShapeMismatchError("executed_steps exceed the previous action horizon")

    target = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
    hard_mask = torch.zeros((batch_size, horizon, 1), device=device, dtype=torch.bool)
    soft_weights = torch.zeros((batch_size, horizon), device=device, dtype=dtype)
    previous = previous.to(device=device, dtype=dtype)
    explicit_weights = None
    if context.overlap_weights is not None:
        explicit_weights = torch.as_tensor(
            context.overlap_weights, device=device, dtype=dtype
        )
        if explicit_weights.ndim == 1:
            if tuple(explicit_weights.shape) != (previous_horizon,):
                raise ShapeMismatchError(
                    "overlap_weights [H] must align with previous_model_actions"
                )
            explicit_weights = explicit_weights.unsqueeze(0).expand(batch_size, -1)
        elif tuple(explicit_weights.shape) != (batch_size, previous_horizon):
            raise ShapeMismatchError(
                "overlap_weights [B, H] must align with previous_model_actions"
            )
    overlap_limit = (
        horizon if context.overlap_horizon is None else context.overlap_horizon
    )
    hard_lengths: list[int] = []
    overlap_lengths: list[int] = []
    for batch_index in range(batch_size):
        remaining = previous[batch_index, executed_steps[batch_index] :]
        overlap = min(int(remaining.shape[0]), horizon, overlap_limit)
        hard_length = min(requested_hard[batch_index], overlap)
        overlap_lengths.append(overlap)
        hard_lengths.append(hard_length)
        if overlap == 0:
            continue
        target[batch_index, :overlap] = remaining[:overlap]
        if explicit_weights is None:
            position_weights = exponential_position_weights(
                overlap,
                initial_weight=1.0,
                decay=context.guidance_decay,
                device=device,
                dtype=dtype,
            )
        else:
            start = executed_steps[batch_index]
            position_weights = explicit_weights[batch_index, start : start + overlap]
        soft_weights[batch_index, :overlap] = position_weights
        if hard_length:
            hard_mask[batch_index, :hard_length] = True
            soft_weights[batch_index, :hard_length] = 0

    return _PreparedRTC(
        target=target,
        hard_mask=hard_mask,
        soft_weights=soft_weights,
        executed_steps=executed_steps,
        hard_prefix_lengths=tuple(hard_lengths),
        overlap_lengths=tuple(overlap_lengths),
    )


def masked_mean_prefix_feature(
    prefix_output: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    *,
    expected_dim: int | None = 2048,
) -> torch.Tensor:
    """Return one masked-mean feature without another OpenPI prefix forward."""

    if not isinstance(prefix_output, torch.Tensor) or prefix_output.ndim != 3:
        raise ShapeMismatchError("prefix_output must be [B, prefix_tokens, hidden_dim]")
    if not isinstance(prefix_pad_masks, torch.Tensor) or prefix_pad_masks.ndim != 2:
        raise ShapeMismatchError("prefix_pad_masks must be [B, prefix_tokens]")
    if prefix_output.shape[:2] != prefix_pad_masks.shape:
        raise ShapeMismatchError("prefix output and mask token shapes differ")
    hidden_dim = int(prefix_output.shape[2])
    if expected_dim is not None and hidden_dim != expected_dim:
        raise ShapeMismatchError(
            f"expected a {expected_dim}-D Pi0.5 prefix feature, got {hidden_dim}"
        )
    mask = prefix_pad_masks.to(device=prefix_output.device, dtype=torch.bool)
    valid_counts = mask.sum(dim=1)
    if bool((valid_counts == 0).any().item()):
        raise ShapeMismatchError("each OpenPI prefix must contain a valid token")
    mask_f32 = mask.to(dtype=torch.float32).unsqueeze(-1)
    return (prefix_output.to(dtype=torch.float32) * mask_f32).sum(dim=1) / (
        valid_counts.to(dtype=torch.float32).unsqueeze(-1)
    )


class OpenPINativeRTCAdapter:
    """Run release/v0.2 OpenPI inference with native denoiser VJP guidance."""

    def __init__(self, model: Any, *, expected_prefix_feature_dim: int | None = 2048):
        self.model = model
        self.expected_prefix_feature_dim = expected_prefix_feature_dim

    def _build_prefix_cache(
        self, observation: Any
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
        str,
    ]:
        model = self.model
        images, img_masks, lang_tokens, lang_masks, state = (
            model._preprocess_observation(observation, train=False)
        )
        model_builder = getattr(model, "_build_prefix_cache", None)
        if callable(model_builder):
            prefix_output, prefix_pad_masks, past_key_values = model_builder(
                images, img_masks, lang_tokens, lang_masks
            )
            return (
                state,
                prefix_output,
                prefix_pad_masks,
                past_key_values,
                "model._build_prefix_cache",
            )

        # release/v0.2 has no public cache helper. Keep this fallback byte-for-
        # byte equivalent in structure to its sample_actions prefix block.
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
            "eager"
        )
        (prefix_output, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return (
            state,
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            "release_v0.2_inline_prefix_cache",
        )

    def _velocity(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        get_velocity = getattr(self.model, "get_velocity", None)
        if callable(get_velocity):
            velocity, suffix_out = get_velocity(
                state,
                x_t,
                timestep,
                prefix_pad_masks,
                past_key_values,
            )
            velocity_path = "model.get_velocity"
        else:
            suffix_out = self.model.get_suffix_out(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
            )
            velocity = self.model.action_out_proj(suffix_out)
            velocity_path = "release_v0.2.get_suffix_out+action_out_proj"
        if not isinstance(velocity, torch.Tensor) or velocity.shape != x_t.shape:
            raise ShapeMismatchError("OpenPI velocity must match [B, H, A_model]")
        if not isinstance(suffix_out, torch.Tensor):
            raise TypeError("OpenPI suffix output must be a torch.Tensor")
        return velocity, suffix_out, velocity_path

    def _suffix_value(
        self,
        suffix_out: torch.Tensor,
        *,
        compute_values: bool,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        config = self.model.config
        if not (
            getattr(config, "add_value_head", False)
            and compute_values
            and not getattr(config, "value_after_vlm", False)
        ):
            return torch.zeros(batch_size, device=device, dtype=torch.float32)
        with torch.no_grad():
            suffix_value = suffix_out.detach()
            if getattr(config, "chunk_critic_input", False):
                suffix_value = suffix_value[:, : config.action_chunk].mean(dim=1)
            else:
                suffix_value = suffix_value.mean(dim=1)
            return self.model.value_head(suffix_value)[:, 0].to(dtype=torch.float32)

    def sample_actions(
        self,
        observation: Any,
        rtc_context: OpenPIRTCContext | None = None,
        *,
        noise: torch.Tensor | None = None,
        mode: str = "eval",
        compute_values: bool = True,
    ) -> dict[str, Any]:
        """Sample one OpenPI chunk and reuse its prefix output as a feature.

        Passing ``rtc_context`` enables native RTC. Real RTC is intentionally
        evaluation-only: training-time SDE/log-prob semantics are not silently
        reused for a VJP-modified transition. With no context this is ordinary
        release/v0.2 evaluation sampling plus a same-forward prefix feature.
        """

        if mode != "eval":
            raise ValueError("native OpenPI RTC sampling is evaluation-only")
        if rtc_context is not None and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "native OpenPI RTC needs autograd; call outside torch.inference_mode()"
            )
        if not hasattr(observation, "state") or not isinstance(
            observation.state, torch.Tensor
        ):
            raise TypeError("OpenPI observation.state must be a torch.Tensor")

        model = self.model
        config = model.config
        batch_size = int(observation.state.shape[0])
        device = observation.state.device
        horizon = int(config.action_horizon)
        action_dim = int(config.action_dim)
        denoise_steps = int(config.num_steps)
        if denoise_steps <= 0:
            raise ValueError("OpenPI num_steps must be positive")
        expected_shape = (batch_size, horizon, action_dim)
        if noise is None:
            with torch.no_grad():
                noise = model.sample_noise(expected_shape, device)
        elif hasattr(model, "action_in_proj") and hasattr(
            model.action_in_proj, "weight"
        ):
            noise = noise.to(device=device, dtype=model.action_in_proj.weight.dtype)
        else:
            noise = noise.to(device=device)
        if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != expected_shape:
            raise ShapeMismatchError(
                f"OpenPI noise must have shape {expected_shape}, got "
                f"{getattr(noise, 'shape', None)}"
            )
        if not torch.is_floating_point(noise):
            raise TypeError("OpenPI noise must use a floating dtype")
        if not bool(torch.isfinite(noise).all().item()):
            raise ValueError("OpenPI noise contains NaN or infinity")

        if rtc_context is None:
            prepared = _PreparedRTC(
                target=torch.zeros(expected_shape, device=device, dtype=noise.dtype),
                hard_mask=torch.zeros(
                    (batch_size, horizon, 1), device=device, dtype=torch.bool
                ),
                soft_weights=torch.zeros(
                    (batch_size, horizon), device=device, dtype=noise.dtype
                ),
                executed_steps=(0,) * batch_size,
                hard_prefix_lengths=(0,) * batch_size,
                overlap_lengths=(0,) * batch_size,
            )
        else:
            prepared = _prepare_rtc(
                rtc_context,
                batch_size=batch_size,
                horizon=horizon,
                action_dim=action_dim,
                device=device,
                dtype=noise.dtype,
            )
        with torch.no_grad():
            (
                state,
                prefix_output,
                prefix_pad_masks,
                past_key_values,
                prefix_cache_path,
            ) = self._build_prefix_cache(observation)
            prefix_feature = masked_mean_prefix_feature(
                prefix_output,
                prefix_pad_masks,
                expected_dim=self.expected_prefix_feature_dim,
            ).detach()
            if getattr(model, "use_vlm_value", False):
                values_vlm = model.get_value_from_vlm(prefix_output).to(
                    dtype=torch.float32
                )

        hard_mask = prepared.hard_mask
        target = prepared.target
        soft_weights = prepared.soft_weights

        def freeze_prefix(actions: torch.Tensor) -> torch.Tensor:
            return torch.where(hard_mask, target, actions)

        x_t = freeze_prefix(noise.detach())
        chains = [x_t.detach()]
        values: list[torch.Tensor] = []
        guidance_losses: list[float] = []
        velocity_paths: set[str] = set()
        vjp_steps = 0
        has_soft_guidance = bool(
            rtc_context is not None
            and rtc_context.guidance_scale > 0
            and torch.any(soft_weights > 0).item()
        )
        # This is exactly release/v0.2's [1, ..., 1/N, 0] Euler schedule.
        timesteps = torch.linspace(
            1.0,
            0.0,
            denoise_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        for step in range(denoise_steps):
            timestep = timesteps[step]
            dt = timesteps[step + 1] - timestep
            timestep_batch = timestep.expand(batch_size)
            x_t = freeze_prefix(x_t)
            if has_soft_guidance:
                with torch.enable_grad():
                    current = x_t.detach().requires_grad_(True)
                    velocity, suffix_out, velocity_path = self._velocity(
                        state,
                        current,
                        timestep_batch,
                        prefix_pad_masks,
                        past_key_values,
                    )
                    clean_estimate = current - timestep * velocity
                    error = clean_estimate - target
                    weighted_error = error.square().sum(dim=-1) * soft_weights
                    denominator = (soft_weights.sum() * float(action_dim)).clamp_min(
                        torch.finfo(noise.dtype).eps
                    )
                    loss = 0.5 * weighted_error.sum() / denominator
                    gradient = torch.autograd.grad(loss, current)[0]
                    gradient = gradient * (soft_weights > 0).unsqueeze(-1)
                    # dt is negative, so +gradient in velocity gives a descent
                    # step in x: x_next = x + dt * (v + scale * grad L).
                    guided_velocity = velocity + rtc_context.guidance_scale * gradient
                    x_next = current + dt * guided_velocity
                    guidance_losses.append(float(loss.detach().cpu().item()))
                    vjp_steps += 1
            else:
                with torch.no_grad():
                    velocity, suffix_out, velocity_path = self._velocity(
                        state,
                        x_t,
                        timestep_batch,
                        prefix_pad_masks,
                        past_key_values,
                    )
                    x_next = x_t + dt * velocity
            velocity_paths.add(velocity_path)
            values.append(
                self._suffix_value(
                    suffix_out,
                    compute_values=compute_values,
                    batch_size=batch_size,
                    device=device,
                )
            )
            # Hard committed actions are exact both at the denoiser input and
            # after every Euler/VJP update.
            x_t = freeze_prefix(x_next).detach()
            chains.append(x_t)

        actions = x_t
        action_chunk = min(int(config.action_chunk), horizon)
        action_env_dim = min(int(config.action_env_dim), action_dim)
        prev_logprobs = torch.zeros(
            (batch_size, action_chunk, action_env_dim),
            device=device,
            dtype=actions.dtype,
        )
        if getattr(model, "use_vlm_value", False):
            prev_values = values_vlm[:, None]
        else:
            prev_values = torch.stack(values, dim=1).mean(dim=1, keepdim=True)
        denoise_inds = torch.full(
            (batch_size, denoise_steps), -1, device=device, dtype=torch.long
        )
        position_weights = tuple(
            tuple(
                float(value)
                for value in soft_weights[batch_index, :overlap].detach().cpu().tolist()
            )
            for batch_index, overlap in enumerate(prepared.overlap_lengths)
        )
        diagnostics = {
            "adapter": (
                "release_v0.2_openpi_native_rtc"
                if rtc_context is not None
                else "release_v0.2_openpi_same_forward_feature"
            ),
            "rtc_native_guidance": rtc_context is not None,
            "rtc_host_postprocess_fallback": False,
            "guidance_mode": (
                "native_denoise_vjp" if rtc_context is not None else "disabled"
            ),
            "model_action_space": True,
            "denoise_steps": denoise_steps,
            "vjp_steps": vjp_steps,
            "guidance_scale": (
                float(rtc_context.guidance_scale) if rtc_context is not None else 0.0
            ),
            "guidance_decay": (
                float(rtc_context.guidance_decay) if rtc_context is not None else None
            ),
            "overlap_weight_source": (
                "explicit"
                if rtc_context is not None and rtc_context.overlap_weights is not None
                else "exponential"
                if rtc_context is not None
                else "disabled"
            ),
            "guidance_losses": tuple(guidance_losses),
            "executed_steps": prepared.executed_steps,
            "hard_prefix_lengths": prepared.hard_prefix_lengths,
            "overlap_lengths": prepared.overlap_lengths,
            "soft_position_weights": position_weights,
            "hard_freeze": "before_denoiser_and_after_euler",
            "flow_convention": "x0=x_t-t*v; x_next=x_t+dt*v",
            "prefix_cache_path": prefix_cache_path,
            "velocity_paths": tuple(sorted(velocity_paths)),
            "prefix_feature_key": OPENPI_RTC_PREFIX_FEATURE_KEY,
            "prefix_feature_dim": int(prefix_feature.shape[-1]),
            "prefix_feature_reused_same_forward": True,
            "real_checkpoint_validated": False,
            "gpu_validated": False,
            "validation_scope": "cpu_unit_only",
        }
        return {
            "actions": actions,
            "chains": torch.stack(chains, dim=1),
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "denoise_inds": denoise_inds,
            OPENPI_RTC_PREFIX_FEATURE_KEY: prefix_feature,
            OPENPI_RTC_DIAGNOSTICS_KEY: diagnostics,
        }

    def sample_actions_with_features(
        self,
        observation: Any,
        *,
        noise: torch.Tensor | None = None,
        mode: str = "eval",
        compute_values: bool = True,
    ) -> dict[str, Any]:
        """Run ordinary eval sampling and return its same-forward prefix feature."""

        return self.sample_actions(
            observation,
            rtc_context=None,
            noise=noise,
            mode=mode,
            compute_values=compute_values,
        )


def _model_observation_from_dict(model: Any, processed_obs: Mapping[str, Any]) -> Any:
    factory = getattr(model, "observation_from_dict", None)
    if callable(factory):
        return factory(processed_obs)
    from openpi.models import model as openpi_model

    return openpi_model.Observation.from_dict(processed_obs)


def predict_action_batch_with_openpi_features(
    model: Any,
    env_observation: Mapping[str, Any],
    *,
    rtc_context: OpenPIRTCContext | None = None,
    compute_values: bool = False,
    expected_prefix_feature_dim: int | None = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run RLinf's eval transforms and one prefix/sampling forward.

    This is the eval-only counterpart of ``model.predict_action_batch``. It
    deliberately calls the model's own observation, input, precision, and
    output transforms. The only duplicated orchestration is necessary because
    release/v0.2 discards ``prefix_output`` before returning from
    ``predict_action_batch``. Normalized ``OPENPI_MODEL_ACTIONS_KEY`` values
    are returned alongside transformed environment actions so native RTC never
    attempts to invert ``output_transform``.

    The caller may use ``torch.inference_mode`` only when ``rtc_context`` is
    ``None``. Native RTC needs autograd for its denoiser VJP.
    """

    to_process_obs = model.obs_processor(dict(env_observation))
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = _model_observation_from_dict(model, processed_obs)
    adapter = OpenPINativeRTCAdapter(
        model,
        expected_prefix_feature_dim=expected_prefix_feature_dim,
    )
    if rtc_context is None:
        outputs = adapter.sample_actions_with_features(
            observation,
            mode="eval",
            compute_values=compute_values,
        )
    else:
        outputs = adapter.sample_actions(
            observation,
            rtc_context,
            mode="eval",
            compute_values=compute_values,
        )

    transformed = model.output_transform(
        {"actions": outputs["actions"], "state": observation.state}
    )
    transformed_actions = transformed["actions"]
    if isinstance(transformed_actions, torch.Tensor):
        actions = transformed_actions.detach().cpu().numpy()
    else:
        actions = np.asarray(transformed_actions)
    if actions.ndim != 3:
        raise ShapeMismatchError(
            f"OpenPI output_transform actions must be [B, H, A], got {actions.shape}"
        )
    model_actions = outputs["actions"][:, : actions.shape[1]].detach()
    forward_inputs: dict[str, Any] = {
        "chains": outputs["chains"],
        "denoise_inds": outputs["denoise_inds"],
    }
    for key in ("tokenized_prompt", "tokenized_prompt_mask"):
        if key in processed_obs:
            forward_inputs[key] = processed_obs[key]
    return np.asarray(actions, dtype=np.float32), {
        "prev_logprobs": outputs["prev_logprobs"],
        "prev_values": outputs["prev_values"],
        "forward_inputs": forward_inputs,
        OPENPI_MODEL_ACTIONS_KEY: model_actions,
        OPENPI_RTC_PREFIX_FEATURE_KEY: outputs[OPENPI_RTC_PREFIX_FEATURE_KEY],
        OPENPI_RTC_DIAGNOSTICS_KEY: outputs[OPENPI_RTC_DIAGNOSTICS_KEY],
    }


def sample_actions_with_openpi_rtc(
    model: Any,
    observation: Any,
    rtc_context: OpenPIRTCContext,
    *,
    noise: torch.Tensor | None = None,
    mode: str = "eval",
    compute_values: bool = True,
    expected_prefix_feature_dim: int | None = 2048,
) -> dict[str, Any]:
    """Functional entry point for :class:`OpenPINativeRTCAdapter`."""

    return OpenPINativeRTCAdapter(
        model,
        expected_prefix_feature_dim=expected_prefix_feature_dim,
    ).sample_actions(
        observation,
        rtc_context,
        noise=noise,
        mode=mode,
        compute_values=compute_values,
    )


__all__ = [
    "OPENPI_MODEL_ACTIONS_KEY",
    "OPENPI_RTC_DIAGNOSTICS_KEY",
    "OPENPI_RTC_PREFIX_FEATURE_KEY",
    "OpenPINativeRTCAdapter",
    "OpenPIRTCContext",
    "masked_mean_prefix_feature",
    "predict_action_batch_with_openpi_features",
    "sample_actions_with_openpi_rtc",
]
