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

"""Validated rollout containers for chunk-level residual post-training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..errors import ShapeMismatchError


@dataclass(frozen=True)
class ResidualRollout:
    """A padded batch of on-policy residual trajectories.

    The first two axes are always ``[batch, chunk_time]``. Reference and
    residual actions retain ``[horizon, action_dim]`` so no action-token
    flattening ambiguity reaches the trainer.
    """

    features: NDArray[np.float32]
    next_features: NDArray[np.float32]
    reference_actions: NDArray[np.float32]
    next_reference_actions: NDArray[np.float32]
    residual_actions: NDArray[np.float32]
    old_log_probs: NDArray[np.float32]
    rewards: NDArray[np.float32]
    terminals: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        arrays = {
            "features": np.asarray(self.features, dtype=np.float32),
            "next_features": np.asarray(self.next_features, dtype=np.float32),
            "reference_actions": np.asarray(self.reference_actions, dtype=np.float32),
            "next_reference_actions": np.asarray(
                self.next_reference_actions, dtype=np.float32
            ),
            "residual_actions": np.asarray(self.residual_actions, dtype=np.float32),
            "old_log_probs": np.asarray(self.old_log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "terminals": np.asarray(self.terminals, dtype=bool),
            "valid_mask": np.asarray(self.valid_mask, dtype=bool),
        }
        batch_time = arrays["rewards"].shape
        if len(batch_time) != 2:
            raise ShapeMismatchError("rewards must be [batch, chunk_time]")
        scalar_fields = ("old_log_probs", "terminals", "valid_mask")
        if any(arrays[name].shape != batch_time for name in scalar_fields):
            raise ShapeMismatchError("scalar rollout fields must share [B, T]")
        feature_shape = arrays["features"].shape
        if len(feature_shape) != 3 or feature_shape[:2] != batch_time:
            raise ShapeMismatchError("features must be [B, T, F]")
        if arrays["next_features"].shape != feature_shape:
            raise ShapeMismatchError("next_features must match features")
        action_shape = arrays["reference_actions"].shape
        if len(action_shape) != 4 or action_shape[:2] != batch_time:
            raise ShapeMismatchError("actions must be [B, T, H, A]")
        for name in ("next_reference_actions", "residual_actions"):
            if arrays[name].shape != action_shape:
                raise ShapeMismatchError(f"{name} must match reference_actions")
        for name, array in arrays.items():
            if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
                raise ShapeMismatchError(f"{name} contains non-finite values")
            object.__setattr__(self, name, array)

    @property
    def batch_size(self) -> int:
        """Number of trajectories."""

        return int(self.rewards.shape[0])

    @property
    def trajectory_length(self) -> int:
        """Maximum number of chunks per trajectory."""

        return int(self.rewards.shape[1])

    @property
    def valid_transitions(self) -> int:
        """Number of non-padding transitions."""

        return int(self.valid_mask.sum())
