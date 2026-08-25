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
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from ..errors import ShapeMismatchError


class ActionSource(IntEnum):
    """Provenance of an executed action chunk.

    Only a stochastic sample from ``POLICY`` may be marked by
    :attr:`ResidualRollout.policy_mask`. All other sources are execution-time
    interventions and therefore have no behavior-policy likelihood for PPO.
    """

    PADDING = 0
    POLICY = 1
    REFERENCE = 2
    GUARD = 3
    CLAMP = 4
    RESCUE = 5


@dataclass(frozen=True)
class ResidualRollout:
    """A padded batch of residual trajectories with explicit provenance.

    The first two axes are always ``[batch, chunk_time]``. Reference and
    residual actions retain ``[horizon, action_dim]`` so no action-token
    flattening ambiguity reaches the trainer. ``old_log_probs`` is finite
    exactly where ``policy_mask`` is true and is NaN elsewhere. This sentinel
    makes it impossible to silently assign policy likelihoods to reference,
    guard, clamp, rescue, deterministic, or padded actions.
    """

    features: NDArray[np.float32]
    next_features: NDArray[np.float32]
    reference_actions: NDArray[np.float32]
    next_reference_actions: NDArray[np.float32]
    residual_actions: NDArray[np.float32]
    old_log_probs: NDArray[np.float32]
    action_source: NDArray[np.int8]
    policy_mask: NDArray[np.bool_]
    rewards: NDArray[np.float32]
    terminals: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        raw_action_source = np.asarray(self.action_source)
        allowed_sources = np.asarray([source.value for source in ActionSource])
        if not np.issubdtype(raw_action_source.dtype, np.integer) or not np.all(
            np.isin(raw_action_source, allowed_sources)
        ):
            raise ShapeMismatchError("action_source contains an unknown value")
        arrays = {
            "features": np.asarray(self.features, dtype=np.float32),
            "next_features": np.asarray(self.next_features, dtype=np.float32),
            "reference_actions": np.asarray(self.reference_actions, dtype=np.float32),
            "next_reference_actions": np.asarray(
                self.next_reference_actions, dtype=np.float32
            ),
            "residual_actions": np.asarray(self.residual_actions, dtype=np.float32),
            "old_log_probs": np.asarray(self.old_log_probs, dtype=np.float32),
            "action_source": raw_action_source.astype(np.int8, copy=False),
            "policy_mask": np.asarray(self.policy_mask, dtype=bool),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "terminals": np.asarray(self.terminals, dtype=bool),
            "valid_mask": np.asarray(self.valid_mask, dtype=bool),
        }
        batch_time = arrays["rewards"].shape
        if len(batch_time) != 2:
            raise ShapeMismatchError("rewards must be [batch, chunk_time]")
        scalar_fields = (
            "old_log_probs",
            "action_source",
            "policy_mask",
            "terminals",
            "valid_mask",
        )
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

        valid = arrays["valid_mask"]
        policy = arrays["policy_mask"]
        sources = arrays["action_source"]
        if np.any(policy & ~valid):
            raise ShapeMismatchError("policy_mask must be a subset of valid_mask")
        if np.any(policy & (sources != ActionSource.POLICY)):
            raise ShapeMismatchError(
                "policy_mask may only select stochastic POLICY actions"
            )
        if np.any(valid & (sources == ActionSource.PADDING)):
            raise ShapeMismatchError("valid transitions cannot use PADDING source")
        if np.any(~valid & (sources != ActionSource.PADDING)):
            raise ShapeMismatchError("invalid transitions must use PADDING source")

        log_probs = arrays["old_log_probs"]
        if np.any(~np.isfinite(log_probs[policy])):
            raise ShapeMismatchError("policy old_log_probs must be finite")
        if np.any(~np.isnan(log_probs[~policy])):
            raise ShapeMismatchError(
                "non-policy old_log_probs must be NaN, not fabricated densities"
            )
        for name, array in arrays.items():
            if (
                name != "old_log_probs"
                and np.issubdtype(array.dtype, np.floating)
                and not np.all(np.isfinite(array))
            ):
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

    @property
    def policy_transitions(self) -> int:
        """Number of stochastic behavior-policy transitions eligible for PPO."""

        return int(self.policy_mask.sum())
