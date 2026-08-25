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

This module imports PyTorch only when explicitly selected. It uses RLinf's
existing ``predict_action_batch`` and data transforms rather than duplicating
π0.5 normalization logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .contracts import ActionChunk, Observation, PolicyOutput
from .errors import ShapeMismatchError


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
            raise ShapeMismatchError(
                f"missing main camera {self.config.main_camera!r}"
            )
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
        """Run RLinf's model-native preprocessing, sampling, and denormalization."""

        env_observation = self._to_env_observation(observation)
        start_ns = time.perf_counter_ns()
        with torch.inference_mode():
            actions, rollout_metadata = self.model.predict_action_batch(
                env_observation, mode="eval", compute_values=False
            )
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1:
            raise ShapeMismatchError(
                f"OpenPI actions must be [1, H, A], got {values.shape}"
            )
        values = values[0]
        if self.config.action_indices is not None:
            values = values[:, np.asarray(self.config.action_indices, dtype=np.int64)]
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return PolicyOutput(
            action=ActionChunk(
                values=values,
                period_s=self.config.period_s,
                source_observation_ns=observation.timestamp_ns,
                metadata={"model_type": "openpi_pi05"},
            ),
            model_latency_ms=latency_ms,
            path="rlinf_openpi_native",
            diagnostics={
                "has_forward_inputs": "forward_inputs" in rollout_metadata,
                "model_action_shape": tuple(values.shape),
            },
        )
