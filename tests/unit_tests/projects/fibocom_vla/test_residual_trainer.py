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

import numpy as np
import torch

from rlinf.projects.fibocom_vla.config import ResidualRLConfig
from rlinf.projects.fibocom_vla.rl.torch_modules import (
    BoundedResidualDistribution,
    ResidualActor,
    TwinQCritic,
)
from rlinf.projects.fibocom_vla.rl.trainer import HybridResidualPPOTrainer
from rlinf.projects.fibocom_vla.rl.trajectory import ResidualRollout


def test_default_actor_meets_declared_parameter_budget() -> None:
    actor = ResidualActor(ResidualRLConfig())
    assert actor.parameter_count == 1_817_310
    assert round(actor.parameter_count / 1_000_000, 2) == 1.82


def test_bounded_distribution_preserves_action_bounds() -> None:
    config = ResidualRLConfig(
        feature_dim=4,
        action_horizon=3,
        action_dim=2,
        actor_hidden_dim=8,
        actor_bottleneck_dim=4,
        target_actor_parameters=184,
        parameter_tolerance=1,
    )
    actor = ResidualActor(config)
    features = torch.zeros(5, 4)
    reference = torch.tensor([[[0.99, -0.99]] * 3] * 5)
    location, log_std = actor(features, reference)
    distribution = BoundedResidualDistribution(location, log_std, reference, config)
    residual, log_prob = distribution.rsample()
    actions = reference + residual
    assert torch.all(actions <= config.action_high)
    assert torch.all(actions >= config.action_low)
    assert torch.isfinite(log_prob).all()


def test_hybrid_trainer_runs_one_finite_update() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    config = ResidualRLConfig(
        feature_dim=4,
        action_horizon=3,
        action_dim=2,
        actor_hidden_dim=8,
        actor_bottleneck_dim=4,
        target_actor_parameters=184,
        parameter_tolerance=1,
    )
    actor = ResidualActor(config)
    critic = TwinQCritic(config, hidden_dims=(16, 8))
    batch_size, trajectory_length = 2, 3
    features = rng.normal(size=(batch_size, trajectory_length, 4)).astype(np.float32)
    next_features = rng.normal(size=features.shape).astype(np.float32)
    reference = np.zeros((batch_size, trajectory_length, 3, 2), dtype=np.float32)
    next_reference = np.zeros_like(reference)
    flat_features = torch.from_numpy(features.reshape(-1, 4))
    flat_reference = torch.from_numpy(reference.reshape(-1, 3, 2))
    with torch.no_grad():
        location, log_std = actor(flat_features, flat_reference)
        distribution = BoundedResidualDistribution(
            location, log_std, flat_reference, config
        )
        residual, old_log_probs = distribution.rsample()
    rollout = ResidualRollout(
        features=features,
        next_features=next_features,
        reference_actions=reference,
        next_reference_actions=next_reference,
        residual_actions=residual.numpy().reshape(batch_size, trajectory_length, 3, 2),
        old_log_probs=old_log_probs.numpy().reshape(batch_size, trajectory_length),
        rewards=np.array([[0, 0, -1], [0, 1, 0]], dtype=np.float32),
        terminals=np.array([[False, False, True], [False, True, False]]),
        valid_mask=np.array([[True, True, True], [True, True, False]]),
    )
    trainer = HybridResidualPPOTrainer(actor, critic, config)
    metrics = trainer.update(rollout)
    assert metrics.valid_transitions == 5
    assert np.isfinite(metrics.actor_loss)
    assert np.isfinite(metrics.critic_loss)

