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

"""Hardware-neutral recorder for provenance-safe residual rollout chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import ResidualRLConfig
from ..errors import ShapeMismatchError
from .archive import ResidualEpisode
from .rewards import ChunkOutcome, truncate_after_second_failure
from .trajectory import ActionSource


def _finite_f32(value, *, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ShapeMismatchError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class RecordedChunk:
    """One transition after execution routing and safety intervention."""

    features: NDArray[np.float32]
    next_features: NDArray[np.float32]
    reference_actions: NDArray[np.float32]
    next_reference_actions: NDArray[np.float32]
    executed_actions: NDArray[np.float32]
    action_source: ActionSource
    outcome: ChunkOutcome
    behavior_log_prob: float | None = None
    stochastic_policy_sample_unchanged: bool = False


class ResidualEpisodeRecorder:
    """Convert executed chunks into one complete residual episode.

    The caller records the action that the robot backend actually accepted,
    not merely the actor proposal. If a guard, clamp, reference route, or
    rescue rewrites a proposal, ``action_source`` must identify that route and
    no PPO likelihood is retained. This keeps intervention data available to
    twin-Q/value learning without treating it as on-policy data.
    """

    def __init__(self, config: ResidualRLConfig) -> None:
        config.validate()
        self.config = config
        self._chunks: list[RecordedChunk] = []

    def append(self, chunk: RecordedChunk) -> None:
        """Validate and append one executed chunk transition."""

        feature_shape = (self.config.feature_dim,)
        action_shape = (self.config.action_horizon, self.config.action_dim)
        features = _finite_f32(chunk.features, name="features")
        next_features = _finite_f32(chunk.next_features, name="next_features")
        reference = _finite_f32(chunk.reference_actions, name="reference_actions")
        next_reference = _finite_f32(
            chunk.next_reference_actions, name="next_reference_actions"
        )
        executed = _finite_f32(chunk.executed_actions, name="executed_actions")
        if features.shape != feature_shape or next_features.shape != feature_shape:
            raise ShapeMismatchError(f"chunk features must have shape {feature_shape}")
        if any(
            value.shape != action_shape
            for value in (reference, next_reference, executed)
        ):
            raise ShapeMismatchError(f"chunk actions must have shape {action_shape}")
        if np.any(reference < self.config.action_low) or np.any(
            reference > self.config.action_high
        ):
            raise ShapeMismatchError("reference action is outside configured bounds")
        if np.any(executed < self.config.action_low) or np.any(
            executed > self.config.action_high
        ):
            raise ShapeMismatchError("executed action is outside configured bounds")

        is_policy = chunk.action_source == ActionSource.POLICY
        eligible = bool(chunk.stochastic_policy_sample_unchanged)
        if eligible and not is_policy:
            raise ValueError("only an unchanged stochastic POLICY sample can enter PPO")
        if eligible:
            if chunk.behavior_log_prob is None or not np.isfinite(
                chunk.behavior_log_prob
            ):
                raise ValueError("an eligible policy sample needs a finite log-prob")
        elif chunk.behavior_log_prob is not None:
            raise ValueError(
                "non-policy or rewritten actions must not retain a behavior log-prob"
            )
        self._chunks.append(
            RecordedChunk(
                features=features,
                next_features=next_features,
                reference_actions=reference,
                next_reference_actions=next_reference,
                executed_actions=executed,
                action_source=chunk.action_source,
                outcome=chunk.outcome,
                behavior_log_prob=chunk.behavior_log_prob,
                stochastic_policy_sample_unchanged=eligible,
            )
        )

    def finish(self) -> ResidualEpisode:
        """Apply the second-failure rule and finalize a terminal episode."""

        if not self._chunks:
            raise ValueError("cannot finish an empty residual episode")
        binary_rewards = np.asarray(
            [chunk.outcome.binary_reward for chunk in self._chunks],
            dtype=np.float32,
        )
        failures = np.asarray(
            [chunk.outcome.failure_event for chunk in self._chunks], dtype=bool
        )
        transformed = truncate_after_second_failure(
            binary_rewards,
            failures,
            terminal_penalty=self.config.second_failure_penalty,
        )
        valid_indices = np.flatnonzero(transformed.valid_mask)
        if not valid_indices.size:
            raise RuntimeError("second-failure transform removed every transition")
        length = int(valid_indices[-1]) + 1
        selected = self._chunks[:length]
        terminals = transformed.terminals[:length].copy()
        if not terminals.any():
            terminals[-1] = True
        sources = np.asarray([chunk.action_source for chunk in selected], dtype=np.int8)
        policy_mask = np.asarray(
            [chunk.stochastic_policy_sample_unchanged for chunk in selected],
            dtype=bool,
        )
        log_probs = np.full(length, np.nan, dtype=np.float32)
        for index, chunk in enumerate(selected):
            if policy_mask[index]:
                log_probs[index] = float(chunk.behavior_log_prob)
        return ResidualEpisode(
            features=np.stack([chunk.features for chunk in selected]),
            next_features=np.stack([chunk.next_features for chunk in selected]),
            reference_actions=np.stack([chunk.reference_actions for chunk in selected]),
            next_reference_actions=np.stack(
                [chunk.next_reference_actions for chunk in selected]
            ),
            residual_actions=np.stack(
                [chunk.executed_actions - chunk.reference_actions for chunk in selected]
            ),
            old_log_probs=log_probs,
            action_source=sources,
            policy_mask=policy_mask,
            rewards=transformed.rewards[:length],
            terminals=terminals,
        )

    @property
    def chunk_count(self) -> int:
        """Number of transitions recorded before terminal truncation."""

        return len(self._chunks)
