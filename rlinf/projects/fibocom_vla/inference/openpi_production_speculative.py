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

"""Fail-closed production speculative policy for the verified pi0.5 stack.

This is intentionally separate from the generic ``SpeculativeChunkPolicy``.
The production path accepts only the three checkpoint-bound concrete objects
that preserve OpenPI model space end to end:

* :class:`VerifiedOpenPITarget` owns the exact main checkpoint and transforms;
* :class:`Pi05DraftProductionAdapter` issues target-signed 32-D Draft output;
* :class:`RLinfOpenPiChunkPolicy` owns observation and output transforms.

One observation is transformed to the real OpenPI ``Observation``, prefetched
once, drafted, and verified with one shared-noise ``B*K`` target call.  Prefix
acceptance is scored on the twelve arm joints only.  Stitching remains in the
complete 32-D normalized model space; both grippers always come from the main
candidate, and only then does the main checkpoint's state-dependent
``output_transform`` produce the 14-D robot command.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ..config import SpeculativeConfig
from ..contracts import ActionChunk, Observation, PolicyOutput
from ..errors import ConfigurationError, ShapeMismatchError
from ..openpi_adapter import RLinfOpenPiChunkPolicy
from .openpi_speculative import (
    OpenPIParallelVerification,
    OpenPIParallelVerifier,
    VerifiedDraftPrediction,
    VerifiedOpenPITarget,
)
from .pi05_draft import Pi05DraftProductionAdapter
from .triton_speculative import (
    Backend,
    TritonSpeculativeConfig,
    TritonSpeculativeResult,
    speculative_verify_and_stitch,
)

_ACTION_HORIZON = 50
_MODEL_ACTION_DIM = 32
_ENV_ACTION_DIM = 14
_ENV_ACTION_INDICES = tuple(range(_ENV_ACTION_DIM))
_ENV_GRIPPER_INDICES = (6, 13)
_ENV_JOINT_INDICES = tuple(
    index for index in range(_ENV_ACTION_DIM) if index not in _ENV_GRIPPER_INDICES
)
_SUPPORTED_VELOCITY_PATHS = frozenset(
    {
        "model.get_velocity",
        "model.denoise_step",
        "model.get_suffix_out+action_out_proj",
    }
)


def _episode_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("episode_id must be a non-empty string")
    return value.strip()


def _finite_tensor(value: Any, name: str, *, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ShapeMismatchError(f"{name} must be a rank-{ndim} Torch tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains NaN or infinity")
    return value


class OpenPIProductionSpeculativePolicy:
    """Run signed pi0.5 Draft verification without generic callable escape hatches."""

    # Speculative prefix selection and host-side stitching are not native
    # denoising-time RTC.  A factory/controller must never infer otherwise.
    rtc_native_supported = False

    def __init__(
        self,
        target: VerifiedOpenPITarget,
        draft: Pi05DraftProductionAdapter,
        main_policy: RLinfOpenPiChunkPolicy,
        config: SpeculativeConfig,
        *,
        episode_id: str,
        verification_times: Sequence[float] = (0.10, 0.05),
        backend: Backend = "auto",
    ) -> None:
        if type(target) is not VerifiedOpenPITarget:
            raise TypeError("target must be an exact VerifiedOpenPITarget")
        if type(draft) is not Pi05DraftProductionAdapter:
            raise TypeError("draft must be an exact Pi05DraftProductionAdapter")
        if type(main_policy) is not RLinfOpenPiChunkPolicy:
            raise TypeError("main_policy must be an exact RLinfOpenPiChunkPolicy")
        if type(config) is not SpeculativeConfig:
            raise TypeError("config must be an exact SpeculativeConfig")
        config.validate()
        if not config.enabled:
            raise ConfigurationError(
                "OpenPI production speculative policy requires config.enabled=true"
            )
        if backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        times = tuple(float(value) for value in verification_times)
        if not times or any(
            not math.isfinite(value) or value <= 0 or value > 1 for value in times
        ):
            raise ValueError("verification_times must be finite and lie in (0, 1]")

        self._target = target
        self._draft = draft
        self._main_policy = main_policy
        self._config = config
        self._verification_times = times
        self._backend: Backend = backend
        self._episode_id = _episode_id(episode_id)
        self._predict_lock = threading.Lock()
        self._previous_model_actions: torch.Tensor | None = None

        # Deliberately pass only the verified target.  In particular, this
        # policy exposes no velocity_target/callable injection surface.
        self._verifier = OpenPIParallelVerifier(target)
        self._verifier_identity = self._verifier
        self._model_identity = target.model
        self._target_contract_identity = target.contract
        self._draft_head_identity = draft.head
        self._main_config_identity = main_policy.config
        self._validate_live_bindings()
        self._verifier.begin_episode(self._episode_id)

    @property
    def episode_id(self) -> str:
        """Return the active episode identifier."""

        return self._episode_id

    @property
    def previous_model_actions(self) -> torch.Tensor | None:
        """Return a defensive copy of the previous normalized 32-D chunk."""

        if not self._predict_lock.acquire(blocking=False):
            raise RuntimeError("speculative policy is busy in another thread")
        try:
            if self._previous_model_actions is None:
                return None
            return self._previous_model_actions.clone()
        finally:
            self._predict_lock.release()

    def _validate_live_bindings(self) -> None:
        """Recheck object identity, manifests, eval mode, and frozen parameters."""

        if (
            self._verifier is not self._verifier_identity
            or self._target.model is not self._model_identity
            or self._target.contract is not self._target_contract_identity
            or self._main_policy.model is not self._model_identity
            or self._main_policy.config is not self._main_config_identity
            or self._draft.head is not self._draft_head_identity
        ):
            raise ConfigurationError(
                "OpenPI production speculative object identity changed after binding"
            )
        if getattr(self._verifier, "verified_target", None) is not self._target:
            raise ConfigurationError("verifier target identity changed after binding")
        if getattr(self._verifier, "model", None) is not self._model_identity:
            raise ConfigurationError("verifier model identity changed after binding")
        if getattr(self._verifier, "_velocity_target", None) is not None:
            raise ConfigurationError(
                "production verifier must not carry an injected velocity callable"
            )

        self._target.validate_runtime_model()
        contract = self._target.contract
        exact_geometry = {
            "policy_family": (contract.policy_family, "pi0.5"),
            "action_horizon": (contract.action_horizon, _ACTION_HORIZON),
            "model_action_dim": (contract.model_action_dim, _MODEL_ACTION_DIM),
            "env_action_indices": (
                contract.env_action_indices,
                _ENV_ACTION_INDICES,
            ),
            "state_dim": (contract.state_dim, _MODEL_ACTION_DIM),
        }
        mismatches = [
            name
            for name, (actual, expected) in exact_geometry.items()
            if actual != expected
        ]
        if mismatches:
            raise ConfigurationError(
                "production pi0.5 geometry mismatch: " + ", ".join(mismatches)
            )
        if len(contract.camera_keys) != 3:
            raise ConfigurationError(
                "production pi0.5 target must bind main/left-wrist/right-wrist cameras"
            )
        if len(contract.environment_action_semantics) != _ENV_ACTION_DIM:
            raise ConfigurationError(
                "production pi0.5 target must bind all 14 state/action semantics"
            )

        adapter = self._main_policy.config
        observed_cameras = (adapter.main_camera, *adapter.resolved_wrist_cameras)
        if observed_cameras != contract.camera_keys:
            raise ConfigurationError(
                "main policy cameras differ from the verified target contract"
            )
        if adapter.expected_state_names != contract.environment_action_semantics:
            raise ConfigurationError(
                "main policy state semantics differ from the verified target contract"
            )
        if adapter.state_indices is not None:
            raise ConfigurationError(
                "production pi0.5 policy forbids pre-transform state reindexing"
            )
        if adapter.action_indices not in (None, _ENV_ACTION_INDICES):
            raise ConfigurationError(
                "production pi0.5 policy forbids non-identity post-transform action reindexing"
            )

        head = self._draft.head
        require_target = getattr(
            head.target_contract, "require_matches_verified_openpi", None
        )
        if not callable(require_target):
            raise ConfigurationError(
                "production Draft lacks a verified OpenPI target contract"
            )
        require_target(self._target)
        if (
            not bool(getattr(head, "production_compatible", False))
            or bool(getattr(head, "training", True))
            or any(parameter.requires_grad for parameter in head.parameters())
        ):
            raise ConfigurationError(
                "production Draft must remain authorized, eval-only, and frozen"
            )

    def _last_actions_for(self, bundle: Any) -> torch.Tensor:
        expected = (
            bundle.batch_size,
            _ACTION_HORIZON,
            _MODEL_ACTION_DIM,
        )
        if self._previous_model_actions is None:
            return torch.zeros(
                expected,
                device=bundle.device,
                dtype=bundle.normalized_state.dtype,
            )
        previous = self._previous_model_actions
        if tuple(previous.shape) != expected[1:]:
            raise ShapeMismatchError("saved previous model actions are not [50,32]")
        return previous[None].to(
            device=bundle.device,
            dtype=bundle.normalized_state.dtype,
        )

    def _sample_shared_noise(
        self,
        draft_actions: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        sampler = getattr(self._model_identity, "sample_noise", None)
        if not callable(sampler):
            raise ConfigurationError(
                "verified OpenPI target must expose its native sample_noise"
            )
        with torch.no_grad():
            noise = sampler(tuple(draft_actions.shape), device)
        noise = _finite_tensor(noise, "OpenPI shared noise", ndim=3)
        if tuple(noise.shape) != tuple(draft_actions.shape):
            raise ShapeMismatchError("OpenPI shared noise must match Draft [B,50,32]")
        if noise.device != device:
            raise ValueError("OpenPI shared noise is on the wrong device")
        if noise.dtype != draft_actions.dtype:
            raise TypeError(
                "OpenPI shared noise and signed Draft output must use one exact dtype"
            )
        return noise

    def _postprocess(
        self,
        verification: OpenPIParallelVerification,
    ) -> tuple[TritonSpeculativeResult, torch.Tensor]:
        joint_model_indices = tuple(
            verification.env_action_indices[index] for index in _ENV_JOINT_INDICES
        )
        result = speculative_verify_and_stitch(
            verification.draft_model[0],
            verification.shared_noise[0],
            verification.predicted_velocity_model[0],
            verification.verification_times,
            TritonSpeculativeConfig(
                absolute_error_threshold=self._config.absolute_error_threshold,
                relative_error_threshold=self._config.relative_error_threshold,
                distance_indices=joint_model_indices,
                evaluation_horizon=_ACTION_HORIZON,
            ),
            backend=self._backend,
        )
        if (
            result.distance_indices != joint_model_indices
            or len(result.distance_indices) != 12
        ):
            raise ConfigurationError("RMS verification did not use exactly 12 joints")

        # The postprocessor stitches all 32 normalized dimensions.  Grippers
        # are safety-critical absolute coordinates, so neither Draft value may
        # survive even when the joint prefix is accepted.
        stitched = result.stitched_actions.clone()
        gripper_model_indices = tuple(
            verification.env_action_indices[index] for index in _ENV_GRIPPER_INDICES
        )
        for model_index in gripper_model_indices:
            stitched[:, model_index] = result.main_candidate[:, model_index]
        return result, stitched

    def _transform_model_actions(
        self,
        model_observation: Any,
        model_actions: torch.Tensor,
    ) -> np.ndarray:
        model_actions = _finite_tensor(model_actions, "stitched model actions", ndim=2)
        if tuple(model_actions.shape) != (_ACTION_HORIZON, _MODEL_ACTION_DIM):
            raise ShapeMismatchError("stitched model actions must be [50,32]")
        state = getattr(model_observation, "state", None)
        if not isinstance(state, torch.Tensor) or tuple(state.shape) != (1, 32):
            raise ShapeMismatchError("OpenPI model observation state must be [1,32]")
        transformed = self._model_identity.output_transform(
            {"actions": model_actions[None], "state": state}
        )
        if not isinstance(transformed, Mapping) or "actions" not in transformed:
            raise TypeError("OpenPI output_transform must return an actions mapping")
        values = transformed["actions"]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 3 or tuple(values.shape[:2]) != (1, _ACTION_HORIZON):
            raise ShapeMismatchError(
                "OpenPI output_transform actions must be [1,50,A_env]"
            )
        values = values[0]
        action_indices = self._main_policy.config.action_indices
        if action_indices is not None:
            values = values[:, np.asarray(action_indices, dtype=np.int64)]
        if tuple(values.shape) != (_ACTION_HORIZON, _ENV_ACTION_DIM):
            raise ShapeMismatchError(
                "OpenPI production commands must be exactly [50,14]"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("OpenPI transformed actions contain NaN or infinity")
        return values.copy()

    @staticmethod
    def _verification_diagnostics(
        verification: OpenPIParallelVerification,
        result: TritonSpeculativeResult,
    ) -> dict[str, Any]:
        return {
            **verification.diagnostics,
            **result.diagnostics,
            "production_contract": "verified_openpi_pi05_draft_v1",
            "model_space_stitch": "normalized_full_32d",
            "rms_environment_joint_indices": _ENV_JOINT_INDICES,
            "gripper_environment_indices": _ENV_GRIPPER_INDICES,
            "gripper_source": "main_candidate_only",
            "output_transform_owner": "verified_main_openpi_target",
            "rtc_native_supported": False,
        }

    def _commit_previous(self, model_actions: Any) -> None:
        if isinstance(model_actions, np.ndarray):
            model_actions = torch.from_numpy(model_actions)
        model_actions = _finite_tensor(model_actions, "previous model actions", ndim=2)
        if tuple(model_actions.shape) != (_ACTION_HORIZON, _MODEL_ACTION_DIM):
            raise ShapeMismatchError("previous model actions must be [50,32]")
        self._previous_model_actions = (
            model_actions.detach().to(device="cpu", dtype=torch.float32).clone()
        )

    def predict(self, observation: Observation) -> PolicyOutput:
        """Draft, verify, 32-D stitch, and transform one project observation."""

        if not self._predict_lock.acquire(blocking=False):
            raise RuntimeError(
                "OpenPI production speculative predict/reset is not reentrant"
            )
        try:
            start_ns = time.perf_counter_ns()
            self._validate_live_bindings()
            model_observation = self._main_policy.prepare_model_observation(observation)
            bundle = self._verifier.prepare_prefix(
                model_observation,
                episode_id=self._episode_id,
            )
            if bundle.batch_size != 1:
                raise ShapeMismatchError(
                    "project ChunkPolicy supports exactly one observation (B=1)"
                )
            draft_prediction = self._draft.predict_for_target(
                self._target,
                prefix_embs=bundle.prefix_embs,
                prefix_pad_masks=bundle.prefix_pad_masks,
                prefix_att_masks=bundle.prefix_att_masks,
                robot_state=bundle.normalized_state,
                last_actions=self._last_actions_for(bundle),
            )
            if not isinstance(draft_prediction, VerifiedDraftPrediction):
                raise TypeError(
                    "production Draft must return a target-signed VerifiedDraftPrediction"
                )
            draft_actions = _finite_tensor(
                draft_prediction.actions_model,
                "signed Draft model actions",
                ndim=3,
            )
            if tuple(draft_actions.shape) != (1, _ACTION_HORIZON, _MODEL_ACTION_DIM):
                raise ShapeMismatchError("signed Draft model actions must be [1,50,32]")
            if draft_actions.device != bundle.device:
                raise ValueError(
                    "signed Draft output and prefix are on different devices"
                )

            noise = self._sample_shared_noise(draft_actions, device=bundle.device)
            verification_times = torch.tensor(
                self._verification_times,
                device=bundle.device,
                dtype=draft_actions.dtype,
            )
            verification = self._verifier.verify(
                bundle,
                draft_prediction,
                noise,
                verification_times,
            )
            if not isinstance(verification, OpenPIParallelVerification):
                raise TypeError(
                    "OpenPI verifier must return OpenPIParallelVerification"
                )
            if (
                verification.episode_id != self._episode_id
                or verification.target_contract is not self._target.contract
                or verification.draft_model is not draft_prediction.actions_model
                or verification.shared_noise is not noise
            ):
                raise ConfigurationError(
                    "OpenPI verification lineage or tensor identity changed"
                )
            if verification.velocity_path not in _SUPPORTED_VELOCITY_PATHS:
                raise ConfigurationError(
                    "OpenPI verification used an unapproved velocity path"
                )
            if tuple(verification.draft_model.shape) != (
                1,
                _ACTION_HORIZON,
                _MODEL_ACTION_DIM,
            ):
                raise ShapeMismatchError("verification batch must be [1,50,32]")

            result, stitched_model = self._postprocess(verification)
            accepted_prefix = int(result.accepted_prefix)
            acceptance_ratio = accepted_prefix / _ACTION_HORIZON
            low_acceptance = (
                accepted_prefix < self._config.minimum_prefix
                or acceptance_ratio < self._config.fallback_acceptance_ratio
            )
            diagnostics = {
                **self._verification_diagnostics(verification, result),
                "acceptance_ratio": acceptance_ratio,
                "minimum_prefix": self._config.minimum_prefix,
                "fallback_acceptance_ratio": (self._config.fallback_acceptance_ratio),
            }
            if low_acceptance:
                # This is a real same-round full denoising call.  The K-point
                # verifier mean is never mislabeled as a full-main takeover.
                main_output = self._main_policy.predict(observation)
                model_values = main_output.action.model_values
                if model_values is None:
                    raise ShapeMismatchError(
                        "full-main fallback did not preserve 32-D model actions"
                    )
                model_values = np.asarray(model_values, dtype=np.float32)
                if tuple(model_values.shape) != (
                    _ACTION_HORIZON,
                    _MODEL_ACTION_DIM,
                ):
                    raise ShapeMismatchError(
                        "full-main fallback model actions must be [50,32]"
                    )
                if tuple(main_output.action.values.shape) != (
                    _ACTION_HORIZON,
                    _ENV_ACTION_DIM,
                ):
                    raise ShapeMismatchError(
                        "full-main fallback commands must be [50,14]"
                    )
                self._commit_previous(model_values)
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                return PolicyOutput(
                    action=main_output.action,
                    model_latency_ms=elapsed_ms,
                    path="openpi_production_speculative_fallback_full_main",
                    accepted_prefix=accepted_prefix,
                    diagnostics={
                        **main_output.diagnostics,
                        **diagnostics,
                        "fallback_reason": "low_acceptance",
                        "fallback_semantics": "same_round_full_main_denoising",
                        "used_full_fallback": True,
                        "main_model_ms": main_output.model_latency_ms,
                    },
                )

            values = self._transform_model_actions(
                model_observation,
                stitched_model,
            )
            model_values = (
                stitched_model.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            self._commit_previous(model_values)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return PolicyOutput(
                action=ActionChunk(
                    values=values,
                    period_s=self._main_policy.config.period_s,
                    source_observation_ns=observation.timestamp_ns,
                    model_values=model_values.copy(),
                    metadata={
                        "model_type": "openpi_pi05",
                        "openpi_model_values_preserved": True,
                        "openpi_model_action_space": (
                            "normalized_before_output_transform"
                        ),
                        "speculative_production_verified": True,
                    },
                ),
                model_latency_ms=elapsed_ms,
                path="openpi_production_speculative_verified",
                accepted_prefix=accepted_prefix,
                diagnostics={
                    **diagnostics,
                    "fallback_reason": None,
                    "fallback_semantics": "not_triggered",
                    "used_full_fallback": False,
                },
            )
        finally:
            self._predict_lock.release()

    def reset_episode(self, episode_id: str) -> None:
        """Invalidate verifier cache and previous model actions atomically."""

        next_episode = _episode_id(episode_id)
        if not self._predict_lock.acquire(blocking=False):
            raise RuntimeError(
                "OpenPI production speculative predict/reset is not reentrant"
            )
        try:
            self._validate_live_bindings()
            self._verifier.reset_episode(self._episode_id)
            self._verifier.begin_episode(next_episode)
            self._episode_id = next_episode
            self._previous_model_actions = None
        finally:
            self._predict_lock.release()


__all__ = ["OpenPIProductionSpeculativePolicy"]
