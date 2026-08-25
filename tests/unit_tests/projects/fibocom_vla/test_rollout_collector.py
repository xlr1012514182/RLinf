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

from __future__ import annotations

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import ResidualRLConfig
from rlinf.projects.fibocom_vla.rl.collector import (
    RecordedChunk,
    ResidualEpisodeRecorder,
)
from rlinf.projects.fibocom_vla.rl.rewards import ChunkOutcome
from rlinf.projects.fibocom_vla.rl.trajectory import ActionSource


def _config() -> ResidualRLConfig:
    return ResidualRLConfig(
        feature_dim=4,
        action_horizon=3,
        action_dim=2,
        actor_hidden_dim=8,
        actor_bottleneck_dim=4,
        target_actor_parameters=184,
        parameter_tolerance=1,
    )


def _chunk(
    *,
    source: ActionSource,
    failure: bool = False,
    eligible: bool = False,
) -> RecordedChunk:
    return RecordedChunk(
        features=np.zeros(4, dtype=np.float32),
        next_features=np.ones(4, dtype=np.float32),
        reference_actions=np.zeros((3, 2), dtype=np.float32),
        next_reference_actions=np.zeros((3, 2), dtype=np.float32),
        executed_actions=np.full((3, 2), 0.01, dtype=np.float32),
        action_source=source,
        outcome=ChunkOutcome(success=False, failure_event=failure),
        behavior_log_prob=-0.5 if eligible else None,
        stochastic_policy_sample_unchanged=eligible,
    )


def test_recorder_truncates_tail_at_second_failure() -> None:
    recorder = ResidualEpisodeRecorder(_config())
    recorder.append(_chunk(source=ActionSource.POLICY, eligible=True))
    recorder.append(_chunk(source=ActionSource.GUARD, failure=True))
    recorder.append(_chunk(source=ActionSource.REFERENCE))
    recorder.append(_chunk(source=ActionSource.RESCUE, failure=True))
    recorder.append(_chunk(source=ActionSource.POLICY, eligible=True))

    episode = recorder.finish()

    assert episode.length == 4
    assert episode.rewards.tolist() == [0.0, 0.0, 0.0, -1.0]
    assert episode.terminals.tolist() == [False, False, False, True]
    assert episode.policy_mask.tolist() == [True, False, False, False]
    assert np.isnan(episode.old_log_probs[1:]).all()


def test_recorder_rejects_logprob_after_guard_rewrite() -> None:
    recorder = ResidualEpisodeRecorder(_config())
    guarded = _chunk(source=ActionSource.GUARD)

    with pytest.raises(ValueError, match="must not retain"):
        recorder.append(
            RecordedChunk(
                **{
                    **guarded.__dict__,
                    "behavior_log_prob": -0.5,
                }
            )
        )
