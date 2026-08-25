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

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from rlinf.projects.fibocom_vla.config import ResidualRLConfig
from rlinf.projects.fibocom_vla.errors import ShapeMismatchError
from rlinf.projects.fibocom_vla.rl.torch_modules import (
    BoundedResidualDistribution,
    FrozenBaseResidualPolicy,
    ResidualActor,
    StateValueCritic,
    TwinQCritic,
)
from rlinf.projects.fibocom_vla.rl.trainer import HybridResidualPPOTrainer
from rlinf.projects.fibocom_vla.rl.trajectory import ActionSource, ResidualRollout


def _tiny_config() -> ResidualRLConfig:
    return ResidualRLConfig(
        feature_dim=4,
        action_horizon=3,
        action_dim=2,
        actor_hidden_dim=8,
        actor_bottleneck_dim=4,
        target_actor_parameters=184,
        parameter_tolerance=1,
    )


def _make_rollout(
    actor: ResidualActor,
    config: ResidualRLConfig,
    *,
    action_source: np.ndarray,
    policy_mask: np.ndarray,
    valid_mask: np.ndarray,
    seed: int = 7,
) -> ResidualRollout:
    rng = np.random.default_rng(seed)
    batch_size, trajectory_length = valid_mask.shape
    features = rng.normal(
        size=(batch_size, trajectory_length, config.feature_dim)
    ).astype(np.float32)
    next_features = rng.normal(size=features.shape).astype(np.float32)
    action_shape = (
        batch_size,
        trajectory_length,
        config.action_horizon,
        config.action_dim,
    )
    reference = np.zeros(action_shape, dtype=np.float32)
    next_reference = np.zeros_like(reference)
    residual = np.zeros_like(reference)
    old_log_probs = np.full(valid_mask.shape, np.nan, dtype=np.float32)

    flat_policy = np.flatnonzero(policy_mask.reshape(-1))
    if flat_policy.size:
        feature_tensor = torch.from_numpy(
            features.reshape(-1, config.feature_dim)[flat_policy]
        )
        reference_tensor = torch.from_numpy(
            reference.reshape(-1, config.action_horizon, config.action_dim)[flat_policy]
        )
        with torch.no_grad():
            location, log_std = actor(feature_tensor, reference_tensor)
            distribution = BoundedResidualDistribution(
                location, log_std, reference_tensor, config
            )
            sampled_residual, sampled_log_probs = distribution.rsample()
        residual.reshape(-1, config.action_horizon, config.action_dim)[flat_policy] = (
            sampled_residual.numpy()
        )
        old_log_probs.reshape(-1)[flat_policy] = sampled_log_probs.numpy()

    rewards = rng.integers(-1, 2, size=(batch_size, trajectory_length)).astype(
        np.float32
    )
    terminals = np.zeros_like(valid_mask, dtype=bool)
    for batch_index in range(batch_size):
        valid_steps = np.flatnonzero(valid_mask[batch_index])
        if valid_steps.size:
            terminals[batch_index, valid_steps[-1]] = True
    return ResidualRollout(
        features=features,
        next_features=next_features,
        reference_actions=reference,
        next_reference_actions=next_reference,
        residual_actions=residual,
        old_log_probs=old_log_probs,
        action_source=action_source,
        policy_mask=policy_mask,
        rewards=rewards,
        terminals=terminals,
        valid_mask=valid_mask,
    )


def _tiny_trainer(
    actor: ResidualActor,
    critic: TwinQCritic,
    config: ResidualRLConfig,
    *,
    value_critic: StateValueCritic | None = None,
) -> HybridResidualPPOTrainer:
    return HybridResidualPPOTrainer(
        actor,
        critic,
        config,
        value_critic=value_critic or StateValueCritic(config, hidden_dims=(16, 8)),
    )


def test_default_actor_meets_declared_parameter_budget() -> None:
    actor = ResidualActor(ResidualRLConfig())
    assert actor.parameter_count == 1_818_542
    assert round(actor.parameter_count / 1_000_000, 2) == 1.82


def test_so101_actor_meets_declared_parameter_budget() -> None:
    actor = ResidualActor(ResidualRLConfig(action_dim=6, actor_hidden_dim=581))
    assert actor.parameter_count == 1_819_139
    assert round(actor.parameter_count / 1_000_000, 2) == 1.82


def test_bounded_distribution_preserves_action_bounds() -> None:
    config = _tiny_config()
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
    assert torch.allclose(log_prob, distribution.log_prob(residual), atol=1e-4)

    outside_support = residual.clone()
    outside_support[0, 0, 0] = distribution.high[0, 0, 0] + 0.01
    with pytest.raises(ValueError, match="no PPO log-probability"):
        distribution.log_prob(outside_support)


def test_policy_act_marks_only_stochastic_samples_as_ppo_eligible() -> None:
    config = _tiny_config()
    actor = ResidualActor(config)
    base = torch.nn.Linear(1, 1)
    policy = FrozenBaseResidualPolicy(
        base,
        actor,
        base_action_fn=lambda _base, observation: observation["reference"],
        feature_fn=lambda _base, observation: observation["features"],
    )
    observation = {
        "features": torch.zeros(2, config.feature_dim),
        "reference": torch.zeros(2, config.action_horizon, config.action_dim),
    }

    sampled = policy.act(observation)
    sampled_distribution = policy.distribution(
        sampled["features"], sampled["reference_actions"]
    )
    assert sampled["policy_mask"].all()
    assert torch.all(sampled["action_source"] == ActionSource.POLICY)
    assert sampled["old_log_probs"] is sampled["log_probs"]
    assert torch.allclose(
        sampled["old_log_probs"],
        sampled_distribution.log_prob(sampled["residual_actions"]),
        atol=1e-4,
    )

    deterministic = policy.act(observation, deterministic=True)
    assert not deterministic["policy_mask"].any()
    assert torch.isnan(deterministic["old_log_probs"]).all()
    assert torch.all(deterministic["action_source"] == ActionSource.POLICY)


def test_hybrid_trainer_runs_one_finite_update() -> None:
    torch.manual_seed(7)
    config = _tiny_config()
    actor = ResidualActor(config)
    critic = TwinQCritic(config, hidden_dims=(16, 8))
    valid_mask = np.array([[True, True, True], [True, True, False]])
    policy_mask = valid_mask.copy()
    action_source = np.where(
        valid_mask, ActionSource.POLICY, ActionSource.PADDING
    ).astype(np.int8)
    rollout = _make_rollout(
        actor,
        config,
        action_source=action_source,
        policy_mask=policy_mask,
        valid_mask=valid_mask,
    )
    trainer = _tiny_trainer(actor, critic, config)
    metrics = trainer.update(rollout)
    assert metrics.valid_transitions == 5
    assert metrics.policy_transitions == 5
    assert np.isfinite(metrics.actor_loss)
    assert np.isfinite(metrics.critic_loss)
    assert np.isfinite(metrics.value_loss)


def test_reference_guard_clamp_and_rescue_actions_never_enter_ppo() -> None:
    torch.manual_seed(11)
    config = _tiny_config()
    actor = ResidualActor(config)
    critic = TwinQCritic(config, hidden_dims=(16, 8))
    sources = np.array(
        [
            [
                ActionSource.POLICY,
                ActionSource.REFERENCE,
                ActionSource.GUARD,
                ActionSource.CLAMP,
                ActionSource.RESCUE,
            ]
        ],
        dtype=np.int8,
    )
    valid_mask = np.ones((1, 5), dtype=bool)
    policy_mask = np.array([[True, False, False, False, False]])
    rollout = _make_rollout(
        actor,
        config,
        action_source=sources,
        policy_mask=policy_mask,
        valid_mask=valid_mask,
        seed=11,
    )

    metrics = _tiny_trainer(actor, critic, config).update(rollout)

    assert metrics.valid_transitions == 5
    assert metrics.policy_transitions == 1
    assert np.isfinite(metrics.ppo_loss)
    assert np.isnan(rollout.old_log_probs[~rollout.policy_mask]).all()


def test_all_nonpolicy_rollout_updates_critics_but_not_actor() -> None:
    torch.manual_seed(13)
    config = _tiny_config()
    actor = ResidualActor(config)
    critic = TwinQCritic(config, hidden_dims=(16, 8))
    sources = np.array(
        [
            [
                ActionSource.REFERENCE,
                ActionSource.GUARD,
                ActionSource.CLAMP,
                ActionSource.RESCUE,
            ]
        ],
        dtype=np.int8,
    )
    valid_mask = np.ones((1, 4), dtype=bool)
    rollout = _make_rollout(
        actor,
        config,
        action_source=sources,
        policy_mask=np.zeros_like(valid_mask),
        valid_mask=valid_mask,
        seed=13,
    )
    actor_before = {
        name: parameter.detach().clone() for name, parameter in actor.named_parameters()
    }

    metrics = _tiny_trainer(actor, critic, config).update(rollout)

    assert metrics.policy_transitions == 0
    assert metrics.actor_loss == 0.0
    assert metrics.ppo_loss == 0.0
    assert np.isfinite(metrics.critic_loss)
    assert np.isfinite(metrics.value_loss)
    for name, parameter in actor.named_parameters():
        assert torch.equal(parameter, actor_before[name])


def test_fabricated_old_log_probability_is_rejected_before_updates() -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    actor = ResidualActor(config)
    critic = TwinQCritic(config, hidden_dims=(16, 8))
    valid_mask = np.ones((1, 2), dtype=bool)
    rollout = _make_rollout(
        actor,
        config,
        action_source=np.full(valid_mask.shape, ActionSource.POLICY, dtype=np.int8),
        policy_mask=valid_mask.copy(),
        valid_mask=valid_mask,
        seed=17,
    )
    bad_log_probs = rollout.old_log_probs.copy()
    bad_log_probs[rollout.policy_mask] += 0.5
    bad_rollout = replace(rollout, old_log_probs=bad_log_probs)
    trainer = _tiny_trainer(actor, critic, config)
    critic_before = copy.deepcopy(trainer.critic.state_dict())
    value_before = copy.deepcopy(trainer.value_critic.state_dict())

    with pytest.raises(ShapeMismatchError, match="actual sampled bounded"):
        trainer.update(bad_rollout)

    for name, tensor in trainer.critic.state_dict().items():
        assert torch.equal(tensor, critic_before[name])
    for name, tensor in trainer.value_critic.state_dict().items():
        assert torch.equal(tensor, value_before[name])


def test_gae_value_baseline_is_independent_of_twin_q() -> None:
    torch.manual_seed(19)
    config = _tiny_config()
    actor_a = ResidualActor(config)
    actor_b = copy.deepcopy(actor_a)
    critic_a = TwinQCritic(config, hidden_dims=(16, 8))
    critic_b = copy.deepcopy(critic_a)
    value_a = StateValueCritic(config, hidden_dims=(16, 8))
    value_b = copy.deepcopy(value_a)
    trainer_a = _tiny_trainer(actor_a, critic_a, config, value_critic=value_a)
    trainer_b = _tiny_trainer(actor_b, critic_b, config, value_critic=value_b)
    with torch.no_grad():
        for module in (trainer_b.critic, trainer_b.target_critic):
            for parameter in module.parameters():
                parameter.fill_(37.0)

    valid_mask = np.ones((1, 3), dtype=bool)
    rollout = _make_rollout(
        actor_a,
        config,
        action_source=np.full(valid_mask.shape, ActionSource.POLICY, dtype=np.int8),
        policy_mask=valid_mask.copy(),
        valid_mask=valid_mask,
        seed=19,
    )
    advantages_a, returns_a = trainer_a._prepare_advantages(rollout)
    advantages_b, returns_b = trainer_b._prepare_advantages(rollout)

    assert torch.equal(advantages_a, advantages_b)
    assert torch.equal(returns_a, returns_b)


def test_rollout_rejects_nonpolicy_source_in_policy_mask() -> None:
    config = _tiny_config()
    actor = ResidualActor(config)
    valid_mask = np.ones((1, 1), dtype=bool)
    rollout = _make_rollout(
        actor,
        config,
        action_source=np.full(valid_mask.shape, ActionSource.POLICY, dtype=np.int8),
        policy_mask=valid_mask.copy(),
        valid_mask=valid_mask,
    )
    invalid_source = rollout.action_source.copy()
    invalid_source[0, 0] = ActionSource.GUARD

    with pytest.raises(ShapeMismatchError, match="stochastic POLICY"):
        replace(rollout, action_source=invalid_source)
