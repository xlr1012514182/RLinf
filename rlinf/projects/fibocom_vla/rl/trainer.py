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

"""Single-round PPO/GAE update with an explicit auxiliary twin-Q objective."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import ResidualRLConfig
from ..errors import ShapeMismatchError
from .gae import compute_gae, normalize_masked
from .torch_modules import (
    BoundedResidualDistribution,
    ResidualActor,
    StateValueCritic,
    TwinQCritic,
)
from .trajectory import ResidualRollout


@dataclass(frozen=True)
class UpdateMetrics:
    """Scalar diagnostics for one declared single-round update."""

    actor_loss: float
    ppo_loss: float
    anchor_loss: float
    q_policy_loss: float
    critic_loss: float
    value_loss: float
    approximate_kl: float
    clip_fraction: float
    advantage_mean: float
    valid_transitions: int
    policy_transitions: int


def _to_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(array, device=device)


class HybridResidualPPOTrainer:
    """Train a residual actor while keeping the VLA base fully frozen.

    The on-policy actor update is clipped PPO with trajectory-level GAE. Twin-Q
    is trained from the same chunk transitions and contributes a small,
    explicitly weighted policy objective. This is a named hybrid rather than a
    silent claim that PPO and off-policy double-Q are the same algorithm.
    """

    def __init__(
        self,
        actor: ResidualActor,
        critic: TwinQCritic,
        config: ResidualRLConfig,
        *,
        value_critic: StateValueCritic | None = None,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        value_learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
    ) -> None:
        config.validate()
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(critic).eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.config = config
        self.value_critic = (
            StateValueCritic(config) if value_critic is None else value_critic
        ).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(
            actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(), lr=critic_learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_critic.parameters(), lr=value_learning_rate
        )
        self.max_grad_norm = max_grad_norm

    @property
    def device(self) -> torch.device:
        """Device hosting the residual actor."""

        return next(self.actor.parameters()).device

    def _distribution(
        self, features: Tensor, reference: Tensor
    ) -> BoundedResidualDistribution:
        location, log_std = self.actor(features, reference)
        return BoundedResidualDistribution(location, log_std, reference, self.config)

    def _state_values(self, features: Tensor, reference: Tensor) -> Tensor:
        """Return the action-independent ``V(s)`` baseline used by GAE."""

        return self.value_critic(features, reference)

    def _prepare_advantages(self, rollout: ResidualRollout) -> tuple[Tensor, Tensor]:
        device = self.device
        batch_size, trajectory_length = rollout.rewards.shape
        with torch.no_grad():
            features = _to_tensor(rollout.features, device).reshape(
                -1, rollout.features.shape[-1]
            )
            reference = _to_tensor(rollout.reference_actions, device).reshape(
                -1, *rollout.reference_actions.shape[-2:]
            )
            values = self._state_values(features, reference).reshape(
                batch_size, trajectory_length
            )
            last_features = _to_tensor(rollout.next_features[:, -1], device)
            last_reference = _to_tensor(rollout.next_reference_actions[:, -1], device)
            bootstrap = self._state_values(last_features, last_reference)
            value_sequence = torch.cat((values, bootstrap[:, None]), dim=1)
        advantages_np, returns_np = compute_gae(
            rollout.rewards,
            value_sequence.detach().cpu().numpy(),
            rollout.terminals,
            rollout.valid_mask,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        advantages_np = normalize_masked(advantages_np, rollout.policy_mask)
        return _to_tensor(advantages_np, device), _to_tensor(returns_np, device)

    def _polyak_update(self) -> None:
        tau = self.config.target_tau
        with torch.no_grad():
            for target, online in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target.lerp_(online, tau)

    def update(self, rollout: ResidualRollout) -> UpdateMetrics:
        """Run one PPO epoch plus value/twin-Q updates.

        PPO and auxiliary actor-Q gradients use only stochastic ``POLICY``
        transitions. The value and twin-Q critics may learn from every valid
        executed action, including reference/guard/clamp/rescue interventions.
        """

        if rollout.valid_transitions == 0:
            raise ShapeMismatchError("rollout contains no valid transitions")
        advantages, returns = self._prepare_advantages(rollout)
        device = self.device
        valid = _to_tensor(rollout.valid_mask, device).bool().reshape(-1)
        policy = _to_tensor(rollout.policy_mask, device).bool().reshape(-1)
        features_all = _to_tensor(rollout.features, device).reshape(
            -1, rollout.features.shape[-1]
        )
        next_features_all = _to_tensor(rollout.next_features, device).reshape(
            -1, rollout.next_features.shape[-1]
        )
        action_tail = rollout.reference_actions.shape[-2:]
        reference_all = _to_tensor(rollout.reference_actions, device).reshape(
            -1, *action_tail
        )
        next_reference_all = _to_tensor(rollout.next_reference_actions, device).reshape(
            -1, *action_tail
        )
        residual_all = _to_tensor(rollout.residual_actions, device).reshape(
            -1, *action_tail
        )
        features = features_all[valid]
        next_features = next_features_all[valid]
        reference = reference_all[valid]
        next_reference = next_reference_all[valid]
        residual = residual_all[valid]
        rewards = _to_tensor(rollout.rewards, device).reshape(-1)[valid]
        terminals = _to_tensor(rollout.terminals, device).reshape(-1)[valid].float()
        value_returns = returns.reshape(-1)[valid]
        adjusted_action = reference + residual

        policy_features = features_all[policy]
        policy_reference = reference_all[policy]
        policy_residual_samples = residual_all[policy]
        old_log_probs = _to_tensor(rollout.old_log_probs, device).reshape(-1)[policy]
        normalized_advantages = advantages.reshape(-1)[policy]
        if rollout.policy_transitions:
            with torch.no_grad():
                behavior_distribution = self._distribution(
                    policy_features, policy_reference
                )
                recomputed_old_log_probs = behavior_distribution.log_prob(
                    policy_residual_samples
                )
            if not torch.allclose(
                recomputed_old_log_probs,
                old_log_probs,
                rtol=1e-5,
                atol=1e-4,
            ):
                max_error = (recomputed_old_log_probs - old_log_probs).abs().max()
                raise ShapeMismatchError(
                    "old_log_probs do not match the actual sampled bounded "
                    "residual distribution of the current behavior actor; "
                    f"maximum absolute error={float(max_error.cpu()):.6g}"
                )

        with torch.no_grad():
            next_distribution = self._distribution(next_features, next_reference)
            next_action = next_reference + next_distribution.mode()
            target_q1, target_q2 = self.target_critic(
                next_features, next_reference, next_action
            )
            target = rewards + self.config.gamma * (1.0 - terminals) * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = self.critic(features, reference, adjusted_action)
        critic_loss = torch.nn.functional.mse_loss(
            q1, target
        ) + torch.nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (self.config.critic_coefficient * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        predicted_values = self._state_values(features, reference)
        value_loss = torch.nn.functional.mse_loss(predicted_values, value_returns)
        self.value_optimizer.zero_grad(set_to_none=True)
        (self.config.value_coefficient * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_critic.parameters(), self.max_grad_norm
        )
        self.value_optimizer.step()

        zero = torch.zeros((), device=device)
        actor_loss = zero
        ppo_loss = zero
        anchor_loss = zero
        q_policy_loss = zero
        approximate_kl = zero
        clip_fraction = zero
        advantage_mean = zero
        if rollout.policy_transitions:
            distribution = self._distribution(policy_features, policy_reference)
            new_log_probs = distribution.log_prob(policy_residual_samples)
            log_ratio = new_log_probs - old_log_probs
            ratio = log_ratio.exp()
            unclipped = -normalized_advantages * ratio
            clipped = -normalized_advantages * ratio.clamp(
                1.0 - self.config.ppo_clip_ratio,
                1.0 + self.config.ppo_clip_ratio,
            )
            ppo_loss = torch.maximum(unclipped, clipped).mean()
            policy_residual = distribution.mode()
            anchor_loss = (policy_residual / self.config.max_residual).square().mean()
            policy_action = policy_reference + policy_residual
            policy_q1, policy_q2 = self.critic(
                policy_features, policy_reference, policy_action
            )
            q_policy_loss = -torch.minimum(policy_q1, policy_q2).mean()
            actor_loss = (
                ppo_loss
                + self.config.anchor_coefficient * anchor_loss
                + self.config.q_policy_coefficient * q_policy_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    ((ratio - 1.0).abs() > self.config.ppo_clip_ratio).float().mean()
                )
                advantage_mean = normalized_advantages.mean()
        self._polyak_update()

        return UpdateMetrics(
            actor_loss=float(actor_loss.detach().cpu()),
            ppo_loss=float(ppo_loss.detach().cpu()),
            anchor_loss=float(anchor_loss.detach().cpu()),
            q_policy_loss=float(q_policy_loss.detach().cpu()),
            critic_loss=float(critic_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            approximate_kl=float(approximate_kl.cpu()),
            clip_fraction=float(clip_fraction.cpu()),
            advantage_mean=float(advantage_mean.detach().cpu()),
            valid_transitions=rollout.valid_transitions,
            policy_transitions=rollout.policy_transitions,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return all trainable and optimizer state needed for exact resume."""

        return {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "value_critic": self.value_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint and optimizer lineage."""

        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.value_critic.load_state_dict(state["value_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
