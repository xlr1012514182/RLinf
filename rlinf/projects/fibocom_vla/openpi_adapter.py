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

"""RLinf OpenPI/π0.5 inference adapter.

This module imports PyTorch only when explicitly selected. It calls RLinf's
own observation/input/output transforms while retaining the normalized model
chunk and the pooled prefix feature from the same prefix forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .contracts import ActionChunk, Observation, PolicyOutput
from .errors import ShapeMismatchError
from .inference.openpi_rtc import (
    OPENPI_MODEL_ACTIONS_KEY,
    OPENPI_RTC_DIAGNOSTICS_KEY,
    OPENPI_RTC_PREFIX_FEATURE_KEY,
    OpenPIRTCContext,
    predict_action_batch_with_openpi_features,
)
from .inference.rtc import RTCConditioning
from .residual_policy import FrozenReferenceOutput


@dataclass(frozen=True)
class OpenPiAdapterConfig:
    """Observation mapping and control-period settings."""

    main_camera: str = "main"
    wrist_camera: str | None = None
    period_s: float = 0.05
    state_indices: tuple[int, ...] | None = None
    action_indices: tuple[int, ...] | None = None


class RLinfOpenPiChunkPolicy:
    """Expose an initialized RLinf OpenPi π0/π0.5 model as ``ChunkPolicy``."""

    def __init__(self, model: Any, config: OpenPiAdapterConfig) -> None:
        if config.period_s <= 0:
            raise ValueError("period_s must be positive")
        self.model = model
        self.config = config

    def _to_env_observation(self, observation: Observation) -> dict[str, Any]:
        if self.config.main_camera not in observation.images:
            raise ShapeMismatchError(f"missing main camera {self.config.main_camera!r}")
        state = observation.state.joint_positions
        if self.config.state_indices is not None:
            state = state[np.asarray(self.config.state_indices, dtype=np.int64)]
        main_image = np.asarray(observation.images[self.config.main_camera])
        wrist_images = None
        if self.config.wrist_camera is not None:
            if self.config.wrist_camera not in observation.images:
                raise ShapeMismatchError(
                    f"missing wrist camera {self.config.wrist_camera!r}"
                )
            wrist_images = torch.from_numpy(
                np.ascontiguousarray(
                    observation.images[self.config.wrist_camera][None, ...]
                )
            )
        return {
            "main_images": torch.from_numpy(
                np.ascontiguousarray(main_image[None, ...])
            ),
            "wrist_images": wrist_images,
            "states": torch.from_numpy(np.ascontiguousarray(state[None, ...])),
            "task_descriptions": [observation.instruction],
        }

    def predict(self, observation: Observation) -> PolicyOutput:
        """Run ordinary eval sampling with a same-forward pooled feature."""

        return self.predict_with_features(observation).output

    @staticmethod
    def _to_numpy(value: Any, *, name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ShapeMismatchError(f"{name} contains NaN or infinity")
        return array

    def _rtc_context(self, conditioning: RTCConditioning) -> OpenPIRTCContext:
        model_hard = conditioning.model_hard_prefix
        model_overlap = conditioning.model_overlap_target
        if model_hard is None or model_overlap is None:
            raise ShapeMismatchError(
                "native OpenPI RTC requires model-space lineage; the previous "
                "ActionChunk.model_values are unavailable or were invalidated"
            )
        hard_length = int(model_hard.shape[0])
        if hard_length > model_overlap.shape[0]:
            raise ShapeMismatchError("model RTC hard prefix exceeds its overlap")
        if not np.array_equal(model_hard, model_overlap[:hard_length]):
            raise ShapeMismatchError(
                "model RTC hard prefix is not the leading overlap target"
            )
        return OpenPIRTCContext(
            previous_model_actions=torch.from_numpy(model_overlap[None].copy()),
            hard_prefix_steps=hard_length,
            overlap_horizon=int(model_overlap.shape[0]),
            overlap_weights=torch.from_numpy(conditioning.overlap_weights.copy()),
            guidance_scale=1.0,
        )

    def _predict_with_features(
        self,
        observation: Observation,
        conditioning: RTCConditioning | None,
    ) -> FrozenReferenceOutput:
        """Share model-native preprocessing across ordinary and RTC prediction."""

        env_observation = self._to_env_observation(observation)
        start_ns = time.perf_counter_ns()
        if conditioning is None:
            with torch.inference_mode():
                actions, rollout_metadata = predict_action_batch_with_openpi_features(
                    self.model,
                    env_observation,
                    compute_values=False,
                )
        else:
            # Do not enter inference_mode here: native RTC evaluates a VJP
            # through every OpenPI denoising step.
            actions, rollout_metadata = predict_action_batch_with_openpi_features(
                self.model,
                env_observation,
                rtc_context=self._rtc_context(conditioning),
                compute_values=False,
            )

        values = self._to_numpy(actions, name="OpenPI transformed actions")
        if values.ndim != 3 or values.shape[0] != 1:
            raise ShapeMismatchError(
                f"OpenPI actions must be [1, H, A], got {values.shape}"
            )
        values = values[0].copy()
        if self.config.action_indices is not None:
            values = values[:, np.asarray(self.config.action_indices, dtype=np.int64)]
        hard_length = 0 if conditioning is None else conditioning.hard_prefix.shape[0]
        if conditioning is not None:
            if (
                hard_length > values.shape[0]
                or conditioning.hard_prefix.shape[1] != values.shape[1]
            ):
                raise ShapeMismatchError(
                    "RTC environment hard prefix does not match transformed OpenPI actions"
                )
            # output_transform may be state-dependent (for example converting
            # delta actions to absolute commands). The corresponding raw model
            # prefix was already frozen inside the denoiser; reuse the exact
            # previously committed environment commands at the execution edge.
            values[:hard_length] = conditioning.hard_prefix
        model_values = self._to_numpy(
            rollout_metadata[OPENPI_MODEL_ACTIONS_KEY],
            name="OpenPI normalized model actions",
        )
        if model_values.ndim != 3 or model_values.shape[0] != 1:
            raise ShapeMismatchError(
                "OpenPI normalized model actions must be [1, H, A_model]"
            )
        model_values = model_values[0]
        if model_values.shape[0] != values.shape[0]:
            raise ShapeMismatchError(
                "OpenPI model and transformed action horizons are not aligned"
            )
        features = self._to_numpy(
            rollout_metadata[OPENPI_RTC_PREFIX_FEATURE_KEY],
            name="OpenPI pooled prefix feature",
        )
        if features.ndim != 2 or features.shape[0] != 1:
            raise ShapeMismatchError(
                f"OpenPI prefix feature must be [1, F], got {features.shape}"
            )
        features = features[0]
        rtc_diagnostics = dict(rollout_metadata.get(OPENPI_RTC_DIAGNOSTICS_KEY, {}))
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        output = PolicyOutput(
            action=ActionChunk(
                values=values.copy(),
                period_s=self.config.period_s,
                source_observation_ns=observation.timestamp_ns,
                model_values=model_values.copy(),
                committed_prefix=int(hard_length),
                metadata={
                    "model_type": "openpi_pi05",
                    "openpi_model_values_preserved": True,
                    "openpi_model_action_space": "normalized_before_output_transform",
                },
            ),
            model_latency_ms=latency_ms,
            path=(
                "rlinf_openpi_native"
                if conditioning is None
                else "rlinf_openpi_native_rtc_vjp"
            ),
            diagnostics={
                "has_forward_inputs": "forward_inputs" in rollout_metadata,
                "model_action_shape": tuple(values.shape),
                "raw_model_action_shape": tuple(model_values.shape),
                "prefix_feature_dim": int(features.shape[0]),
                "prefix_feature_reused_same_forward": True,
                "committed_env_prefix_reused": conditioning is not None,
                **rtc_diagnostics,
            },
        )
        return FrozenReferenceOutput(output=output, features=features)

    def predict_with_features(self, observation: Observation) -> FrozenReferenceOutput:
        """Return transformed actions and the same-forward 2048-D feature."""

        return self._predict_with_features(observation, conditioning=None)

    def predict_with_rtc(
        self,
        observation: Observation,
        conditioning: RTCConditioning,
    ) -> PolicyOutput:
        """Run native model-space RTC or fail closed when lineage is absent."""

        return self._predict_with_features(observation, conditioning).output
