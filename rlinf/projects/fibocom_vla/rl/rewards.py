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

"""Sparse chunk rewards and the declared second-failure terminal transform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..errors import ShapeMismatchError


@dataclass(frozen=True)
class ChunkOutcome:
    """Task-level binary supervision for one executed action chunk.

    ``success`` is the only positive task reward. ``failure_event`` records a
    discrete execution failure such as a dropped block; it is used by the
    second-failure terminal rule and is not a hand-shaped dense reward.
    """

    success: bool
    failure_event: bool = False

    def __post_init__(self) -> None:
        if self.success and self.failure_event:
            raise ValueError("a chunk cannot be both successful and failed")

    @property
    def binary_reward(self) -> float:
        """Return the unshaped task reward in ``{0, 1}``."""

        return float(self.success)


@dataclass(frozen=True)
class SecondFailureResult:
    """Transformed rewards, masks, and terminal flags."""

    rewards: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    terminals: NDArray[np.bool_]
    second_failure_indices: NDArray[np.int64]


def _ensure_batch_time(array: NDArray, *, name: str) -> tuple[NDArray, bool]:
    if array.ndim == 1:
        return array[None, :], True
    if array.ndim == 2:
        return array, False
    raise ShapeMismatchError(f"{name} must be [T] or [B, T], got {array.shape}")


def truncate_after_second_failure(
    rewards: NDArray[np.floating],
    failure_events: NDArray[np.bool_],
    valid_mask: NDArray[np.bool_] | None = None,
    *,
    terminal_penalty: float = -1.0,
) -> SecondFailureResult:
    """Terminate a trajectory on its second discrete failure.

    The task signal remains chunk-level binary success. At the second failure,
    the transition is made terminal and receives a fixed negative outcome;
    every later padded/tail transition is zeroed and masked out. This makes the
    negative advantage a consequence of a terminal failure outcome rather than
    dense per-step reward shaping.

    Args:
        rewards: Binary success rewards with shape ``[T]`` or ``[B, T]``.
        failure_events: Boolean failure markers with the same shape.
        valid_mask: Optional mask for already padded trajectories.
        terminal_penalty: Outcome assigned to the second failure transition.

    Returns:
        Rewards/masks with the same rank as the input and per-batch indices.
    """

    reward_array, squeezed = _ensure_batch_time(
        np.asarray(rewards, dtype=np.float32), name="rewards"
    )
    failures, failures_squeezed = _ensure_batch_time(
        np.asarray(failure_events, dtype=bool), name="failure_events"
    )
    if squeezed != failures_squeezed or reward_array.shape != failures.shape:
        raise ShapeMismatchError("rewards and failure_events must have equal shapes")
    if not np.all(np.isin(reward_array, (0.0, 1.0))):
        raise ValueError("input task rewards must be binary values in {0, 1}")
    if valid_mask is None:
        mask = np.ones_like(failures, dtype=bool)
    else:
        mask, mask_squeezed = _ensure_batch_time(
            np.asarray(valid_mask, dtype=bool), name="valid_mask"
        )
        if mask_squeezed != squeezed or mask.shape != reward_array.shape:
            raise ShapeMismatchError("valid_mask must match rewards")

    transformed = reward_array.copy()
    output_mask = mask.copy()
    terminals = np.zeros_like(mask, dtype=bool)
    second_indices = np.full(reward_array.shape[0], -1, dtype=np.int64)

    for batch_index in range(reward_array.shape[0]):
        failure_indices = np.flatnonzero(failures[batch_index] & mask[batch_index])
        if failure_indices.size < 2:
            continue
        second = int(failure_indices[1])
        second_indices[batch_index] = second
        transformed[batch_index, second] = terminal_penalty
        terminals[batch_index, second] = True
        if second + 1 < reward_array.shape[1]:
            transformed[batch_index, second + 1 :] = 0.0
            output_mask[batch_index, second + 1 :] = False
            terminals[batch_index, second + 1 :] = False

    if squeezed:
        return SecondFailureResult(
            rewards=transformed[0],
            valid_mask=output_mask[0],
            terminals=terminals[0],
            second_failure_indices=second_indices,
        )
    return SecondFailureResult(
        rewards=transformed,
        valid_mask=output_mask,
        terminals=terminals,
        second_failure_indices=second_indices,
    )
