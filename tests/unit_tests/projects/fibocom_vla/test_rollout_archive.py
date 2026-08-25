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

from rlinf.projects.fibocom_vla.rl.archive import (
    ResidualEpisode,
    RolloutArchiveMetadata,
    load_rollout_archive,
    pad_residual_episodes,
    save_rollout_archive,
)
from rlinf.projects.fibocom_vla.rl.trajectory import ActionSource


def _episode(length: int, *, policy: bool) -> ResidualEpisode:
    feature_dim, horizon, action_dim = 4, 3, 2
    source = ActionSource.POLICY if policy else ActionSource.REFERENCE
    return ResidualEpisode(
        features=np.zeros((length, feature_dim), dtype=np.float32),
        next_features=np.ones((length, feature_dim), dtype=np.float32),
        reference_actions=np.zeros((length, horizon, action_dim), dtype=np.float32),
        next_reference_actions=np.zeros(
            (length, horizon, action_dim), dtype=np.float32
        ),
        residual_actions=np.zeros((length, horizon, action_dim), dtype=np.float32),
        old_log_probs=np.full(length, -0.25 if policy else np.nan, dtype=np.float32),
        action_source=np.full(length, source, dtype=np.int8),
        policy_mask=np.full(length, policy, dtype=bool),
        rewards=np.asarray([0.0] * (length - 1) + [1.0], dtype=np.float32),
        terminals=np.asarray([False] * (length - 1) + [True]),
    )


def test_pad_residual_episodes_keeps_padding_out_of_ppo() -> None:
    rollout = pad_residual_episodes(
        [_episode(2, policy=True), _episode(1, policy=False)]
    )

    assert rollout.valid_transitions == 3
    assert rollout.policy_transitions == 2
    assert rollout.action_source[1, 1] == ActionSource.PADDING
    assert np.isnan(rollout.old_log_probs[1, 1])


def test_rollout_archive_round_trip_and_checksum(tmp_path) -> None:
    rollout = pad_residual_episodes(
        [_episode(2, policy=True), _episode(1, policy=False)]
    )
    path = tmp_path / "rollout.npz"
    metadata = RolloutArchiveMetadata(
        task="block stacking",
        base_model="pi0.5",
        base_model_revision="model-sha",
        behavior_policy_revision="actor-sha",
        seed=17,
        configuration_sha256="config-sha",
    )

    digest = save_rollout_archive(path, rollout, metadata)
    loaded, loaded_metadata, manifest = load_rollout_archive(path)

    assert len(digest) == 64
    assert loaded.valid_transitions == 3
    assert loaded.policy_transitions == 2
    assert loaded_metadata.base_model_revision == "model-sha"
    assert manifest["summary"]["episodes"] == 2

    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_rollout_archive(path)
