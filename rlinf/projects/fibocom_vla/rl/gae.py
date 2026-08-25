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

"""Mask-aware trajectory-level generalized advantage estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..errors import ShapeMismatchError


def compute_gae(
    rewards: NDArray[np.floating],
    values: NDArray[np.floating],
    terminals: NDArray[np.bool_],
    valid_mask: NDArray[np.bool_],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute GAE over complete chunk trajectories.

    ``values`` must contain one bootstrap value beyond rewards, i.e.
    ``[B, T + 1]``. Invalid tail entries never bootstrap into earlier valid
    entries. A terminal transition also cuts the bootstrap explicitly.
    """

    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    terminals_array = np.asarray(terminals, dtype=bool)
    mask_array = np.asarray(valid_mask, dtype=bool)
    squeezed = rewards_array.ndim == 1
    if squeezed:
        rewards_array = rewards_array[None, :]
        values_array = values_array[None, :]
        terminals_array = terminals_array[None, :]
        mask_array = mask_array[None, :]
    if rewards_array.ndim != 2:
        raise ShapeMismatchError("rewards must have shape [T] or [B, T]")
    if terminals_array.shape != rewards_array.shape or mask_array.shape != rewards_array.shape:
        raise ShapeMismatchError("terminals and valid_mask must match rewards")
    if values_array.shape != (rewards_array.shape[0], rewards_array.shape[1] + 1):
        raise ShapeMismatchError("values must have shape [B, T + 1]")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    advantages = np.zeros_like(rewards_array, dtype=np.float32)
    running = np.zeros(rewards_array.shape[0], dtype=np.float32)
    for step in range(rewards_array.shape[1] - 1, -1, -1):
        valid = mask_array[:, step].astype(np.float32)
        nonterminal = (~terminals_array[:, step]).astype(np.float32)
        if step + 1 < rewards_array.shape[1]:
            next_valid = mask_array[:, step + 1].astype(np.float32)
        else:
            next_valid = np.ones_like(valid)
        continuation = nonterminal * next_valid
        delta = (
            rewards_array[:, step]
            + gamma * values_array[:, step + 1] * continuation
            - values_array[:, step]
        )
        running = delta + gamma * gae_lambda * continuation * running
        running *= valid
        advantages[:, step] = running
    returns = (advantages + values_array[:, :-1]) * mask_array
    advantages *= mask_array
    if squeezed:
        return advantages[0], returns[0]
    return advantages, returns


def normalize_masked(
    values: NDArray[np.floating],
    valid_mask: NDArray[np.bool_],
    *,
    epsilon: float = 1e-8,
) -> NDArray[np.float32]:
    """Normalize only valid entries and leave padding at zero."""

    array = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if array.shape != mask.shape:
        raise ShapeMismatchError("values and valid_mask must have equal shapes")
    selected = array[mask]
    output = np.zeros_like(array)
    if selected.size == 0:
        return output
    output[mask] = (selected - selected.mean()) / (selected.std() + epsilon)
    return output
