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

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from rlinf.projects.fibocom_vla.config import (
    ResidualRLConfig,
    RTCConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.rl.archive import (
    ResidualEpisode,
    RolloutArchiveMetadata,
    pad_residual_episodes,
    save_rollout_archive,
)
from rlinf.projects.fibocom_vla.rl.checkpoint import (
    file_sha256,
    load_residual_checkpoint,
)
from rlinf.projects.fibocom_vla.rl.torch_modules import (
    BoundedResidualDistribution,
    ResidualActor,
)
from rlinf.projects.fibocom_vla.rl.train_cli import (
    initialize_checkpoint,
    update_checkpoint,
)
from rlinf.projects.fibocom_vla.rl.trajectory import ActionSource


def _tiny_config_file(tmp_path) -> tuple[StackConfig, object]:
    config = StackConfig(
        residual_rl=ResidualRLConfig(
            feature_dim=4,
            action_horizon=3,
            action_dim=2,
            actor_hidden_dim=8,
            actor_bottleneck_dim=4,
            target_actor_parameters=184,
            parameter_tolerance=1,
        ),
        rtc=RTCConfig(execution_horizon=2, overlap_horizon=3),
        robot=replace(
            StackConfig().robot,
            action_dim=2,
            joint_names=("a", "b"),
            joint_lower=(-1.0, -1.0),
            joint_upper=(1.0, 1.0),
            max_step=(0.1, 0.1),
        ),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    return config, path


def _policy_episode(config: StackConfig, checkpoint_path) -> ResidualEpisode:
    checkpoint, _ = load_residual_checkpoint(checkpoint_path)
    actor = ResidualActor(config.residual_rl)
    actor.load_state_dict(checkpoint["trainer_state"]["actor"])
    features = torch.zeros(1, config.residual_rl.feature_dim)
    reference = torch.zeros(
        1,
        config.residual_rl.action_horizon,
        config.residual_rl.action_dim,
    )
    with torch.no_grad():
        location, log_std = actor(features, reference)
        distribution = BoundedResidualDistribution(
            location, log_std, reference, config.residual_rl
        )
        residual, log_prob = distribution.rsample()
    return ResidualEpisode(
        features=features.numpy(),
        next_features=features.numpy(),
        reference_actions=reference.numpy(),
        next_reference_actions=reference.numpy(),
        residual_actions=residual.numpy(),
        old_log_probs=log_prob.numpy(),
        action_source=np.asarray([ActionSource.POLICY], dtype=np.int8),
        policy_mask=np.asarray([True]),
        rewards=np.asarray([1.0], dtype=np.float32),
        terminals=np.asarray([True]),
    )


def test_checkpoint_rollout_update_has_closed_hash_lineage(tmp_path) -> None:
    config, config_path = _tiny_config_file(tmp_path)
    initial_path = tmp_path / "initial.pt"
    initialized = initialize_checkpoint(
        config_path,
        initial_path,
        base_model="pi0.5",
        base_model_revision="base-sha",
        seed=17,
        device="cpu",
    )
    torch.manual_seed(23)
    rollout_path = tmp_path / "rollout.npz"
    save_rollout_archive(
        rollout_path,
        pad_residual_episodes([_policy_episode(config, initial_path)]),
        RolloutArchiveMetadata(
            task="stack",
            base_model="pi0.5",
            base_model_revision="base-sha",
            behavior_policy_revision=initialized["checkpoint_sha256"],
            seed=17,
            configuration_sha256=file_sha256(config_path),
        ),
    )
    output_path = tmp_path / "updated.pt"

    result = update_checkpoint(
        config_path,
        initial_path,
        rollout_path,
        output_path,
        device="cpu",
    )
    updated, _ = load_residual_checkpoint(output_path)

    assert result["parent_checkpoint_sha256"] == initialized["checkpoint_sha256"]
    assert updated["metadata"]["declared_ppo_epochs"] == 1
    assert updated["metadata"]["update_metrics"]["policy_transitions"] == 1


def test_update_rejects_wrong_behavior_checkpoint_hash(tmp_path) -> None:
    config, config_path = _tiny_config_file(tmp_path)
    initial_path = tmp_path / "initial.pt"
    initialize_checkpoint(
        config_path,
        initial_path,
        base_model="pi0.5",
        base_model_revision="base-sha",
        seed=17,
        device="cpu",
    )
    rollout_path = tmp_path / "rollout.npz"
    save_rollout_archive(
        rollout_path,
        pad_residual_episodes([_policy_episode(config, initial_path)]),
        RolloutArchiveMetadata(
            task="stack",
            base_model="pi0.5",
            base_model_revision="base-sha",
            behavior_policy_revision="wrong-sha",
            seed=17,
            configuration_sha256=file_sha256(config_path),
        ),
    )

    with pytest.raises(ValueError, match="behavior_policy_revision"):
        update_checkpoint(
            config_path,
            initial_path,
            rollout_path,
            tmp_path / "updated.pt",
            device="cpu",
        )
