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

"""PyTorch modules for frozen-base, reference-anchored residual control."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..config import ResidualRLConfig
from ..errors import ConfigurationError, ShapeMismatchError
from .trajectory import ActionSource


def trainable_parameter_count(module: nn.Module) -> int:
    """Count trainable scalar parameters."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


class ResidualActor(nn.Module):
    """Parameter-budgeted actor that predicts a local action-chunk residual.

    With the audited ``2048``-D π0.5 prefix-mean feature, ``50 x 7`` action
    chunk, hidden width ``562``, and bottleneck ``512``, this module has
    1,818,542 trainable parameters (including log standard deviation), which
    rounds to 1.82M. The twin-Q and value heads are counted separately.
    """

    def __init__(self, config: ResidualRLConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        action_size = config.action_horizon * config.action_dim
        input_size = config.feature_dim + action_size
        self.network = nn.Sequential(
            nn.Linear(input_size, config.actor_hidden_dim),
            nn.LayerNorm(config.actor_hidden_dim),
            nn.GELU(),
            nn.Linear(config.actor_hidden_dim, config.actor_bottleneck_dim),
            nn.LayerNorm(config.actor_bottleneck_dim),
            nn.GELU(),
            nn.Linear(config.actor_bottleneck_dim, action_size),
        )
        self.log_std = nn.Parameter(
            torch.full((config.action_horizon, config.action_dim), -2.3)
        )
        output_layer = self.network[-1]
        nn.init.normal_(output_layer.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(output_layer.bias)
        count = trainable_parameter_count(self)
        if abs(count - config.target_actor_parameters) > config.parameter_tolerance:
            raise ConfigurationError(
                "residual actor parameter budget mismatch: "
                f"actual={count:,}, target={config.target_actor_parameters:,} "
                f"±{config.parameter_tolerance:,}"
            )

    @property
    def parameter_count(self) -> int:
        """Return the audited trainable parameter count."""

        return trainable_parameter_count(self)

    def forward(
        self, features: Tensor, reference_actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return pre-tanh residual location and broadcast log standard deviation."""

        expected_reference = (
            self.config.action_horizon,
            self.config.action_dim,
        )
        if features.shape[-1] != self.config.feature_dim:
            raise ShapeMismatchError(
                f"expected feature dim {self.config.feature_dim}, got {features.shape}"
            )
        if reference_actions.shape[-2:] != expected_reference:
            raise ShapeMismatchError(
                f"expected action tail {expected_reference}, got {reference_actions.shape}"
            )
        if features.shape[:-1] != reference_actions.shape[:-2]:
            raise ShapeMismatchError("feature and action batch axes differ")
        flat_reference = reference_actions.flatten(start_dim=-2)
        location = self.network(torch.cat((features, flat_reference), dim=-1))
        location = location.unflatten(
            -1, (self.config.action_horizon, self.config.action_dim)
        )
        return location, self.log_std.expand_as(location)


class BoundedResidualDistribution:
    """Tanh-normal residual distribution constrained around a reference action."""

    _EPSILON = 1e-6

    def __init__(
        self,
        location: Tensor,
        log_std: Tensor,
        reference_actions: Tensor,
        config: ResidualRLConfig,
    ) -> None:
        if location.shape != reference_actions.shape or log_std.shape != location.shape:
            raise ShapeMismatchError("residual distribution tensors must share a shape")
        if torch.any(reference_actions < config.action_low) or torch.any(
            reference_actions > config.action_high
        ):
            raise ValueError("reference action lies outside configured action bounds")
        self.location = location
        self.log_std = log_std.clamp(-7.0, 1.0)
        self.reference_actions = reference_actions
        low_limit = torch.full_like(reference_actions, config.action_low)
        high_limit = torch.full_like(reference_actions, config.action_high)
        max_delta = torch.full_like(reference_actions, config.max_residual)
        self.low = torch.maximum(-max_delta, low_limit - reference_actions)
        self.high = torch.minimum(max_delta, high_limit - reference_actions)
        self.midpoint = (self.low + self.high) * 0.5
        self.half_range = ((self.high - self.low) * 0.5).clamp_min(self._EPSILON)
        self.base = torch.distributions.Normal(self.location, self.log_std.exp())

    def _transform(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        unit = torch.tanh(latent)
        residual = self.midpoint + self.half_range * unit
        log_jacobian = torch.log(
            self.half_range * (1.0 - unit.square()) + self._EPSILON
        )
        return residual, log_jacobian

    def rsample(self) -> tuple[Tensor, Tensor]:
        """Reparameterized residual and its chunk-level log probability."""

        latent = self.base.rsample()
        residual, log_jacobian = self._transform(latent)
        log_prob = (self.base.log_prob(latent) - log_jacobian).sum(dim=(-2, -1))
        return residual, log_prob

    def mode(self) -> Tensor:
        """Return the bounded deterministic residual."""

        residual, _ = self._transform(self.location)
        return residual

    def log_prob(self, residual: Tensor) -> Tensor:
        """Evaluate a bounded residual with inverse-tanh correction."""

        if residual.shape != self.location.shape:
            raise ShapeMismatchError("residual must match distribution shape")
        if torch.any(residual < self.low - self._EPSILON) or torch.any(
            residual > self.high + self._EPSILON
        ):
            raise ValueError(
                "residual lies outside this reference-conditioned distribution; "
                "execution-time guard/clamp/rescue actions have no PPO log-probability"
            )
        unit = ((residual - self.midpoint) / self.half_range).clamp(
            -1.0 + self._EPSILON, 1.0 - self._EPSILON
        )
        latent = torch.atanh(unit)
        log_jacobian = torch.log(
            self.half_range * (1.0 - unit.square()) + self._EPSILON
        )
        return (self.base.log_prob(latent) - log_jacobian).sum(dim=(-2, -1))


class _QNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_size: int,
        hidden_dims: tuple[int, int],
    ) -> None:
        super().__init__()
        input_dim = feature_dim + 2 * action_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, features: Tensor, reference: Tensor, action: Tensor) -> Tensor:
        inputs = torch.cat(
            (features, reference.flatten(start_dim=-2), action.flatten(start_dim=-2)),
            dim=-1,
        )
        return self.network(inputs).squeeze(-1)


class TwinQCritic(nn.Module):
    """Independent double-Q critic conditioned on base and adjusted chunks."""

    def __init__(
        self,
        config: ResidualRLConfig,
        hidden_dims: tuple[int, int] = (512, 256),
    ) -> None:
        super().__init__()
        action_size = config.action_horizon * config.action_dim
        self.q1 = _QNetwork(config.feature_dim, action_size, hidden_dims)
        self.q2 = _QNetwork(config.feature_dim, action_size, hidden_dims)

    def forward(
        self, features: Tensor, reference_actions: Tensor, adjusted_actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Evaluate both critics without early minimization."""

        return (
            self.q1(features, reference_actions, adjusted_actions),
            self.q2(features, reference_actions, adjusted_actions),
        )


class StateValueCritic(nn.Module):
    """State-value baseline independent of the residual action and twin-Q.

    PPO/GAE needs an estimate of ``V(s)`` under the behavior policy. Reusing
    ``min Q(s, mode(policy))`` silently substitutes a deterministic action
    value and couples advantages to the auxiliary Q objective. This head is
    conditioned only on cached frozen-base features and the reference chunk,
    making that separation explicit and directly testable.
    """

    def __init__(
        self,
        config: ResidualRLConfig,
        hidden_dims: tuple[int, int] = (512, 256),
    ) -> None:
        super().__init__()
        self.config = config
        action_size = config.action_horizon * config.action_dim
        input_dim = config.feature_dim + action_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, features: Tensor, reference_actions: Tensor) -> Tensor:
        """Estimate ``V(s)`` without reading an executed or actor-mode action."""

        expected_reference = (
            self.config.action_horizon,
            self.config.action_dim,
        )
        if features.shape[-1] != self.config.feature_dim:
            raise ShapeMismatchError(
                f"expected feature dim {self.config.feature_dim}, got {features.shape}"
            )
        if reference_actions.shape[-2:] != expected_reference:
            raise ShapeMismatchError(
                f"expected action tail {expected_reference}, got {reference_actions.shape}"
            )
        if features.shape[:-1] != reference_actions.shape[:-2]:
            raise ShapeMismatchError("feature and reference batch axes differ")
        inputs = torch.cat((features, reference_actions.flatten(start_dim=-2)), dim=-1)
        return self.network(inputs).squeeze(-1)


class FrozenBaseResidualPolicy(nn.Module):
    """Compose an immutable VLA base with the trainable local residual actor.

    ``base_action_fn`` and ``feature_fn`` adapt the concrete π0.5 interface.
    They may accept a dictionary or a model-specific observation object. The
    base forward pass always runs in inference mode without autograd.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        actor: ResidualActor,
        base_action_fn: Callable[[nn.Module, Any], Tensor],
        feature_fn: Callable[[nn.Module, Any], Tensor],
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.actor = actor
        self.base_action_fn = base_action_fn
        self.feature_fn = feature_fn
        self.freeze_base()

    def freeze_base(self) -> None:
        """Disable base gradients and stochastic training behavior."""

        self.base_policy.eval()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

    def assert_base_frozen(self) -> None:
        """Fail fast if any optimizer accidentally re-enabled the VLA base."""

        trainable = [
            name
            for name, value in self.base_policy.named_parameters()
            if value.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"base policy is not frozen: {trainable[:5]}")

    @torch.no_grad()
    def encode_reference(self, observation: Any) -> tuple[Tensor, Tensor]:
        """Return detached base features and reference action chunk."""

        self.assert_base_frozen()
        features = self.feature_fn(self.base_policy, observation).detach()
        reference = self.base_action_fn(self.base_policy, observation).detach()
        return features, reference

    def distribution(
        self, features: Tensor, reference_actions: Tensor
    ) -> BoundedResidualDistribution:
        """Build the trainable residual distribution for cached base outputs."""

        location, log_std = self.actor(features, reference_actions)
        return BoundedResidualDistribution(
            location, log_std, reference_actions, self.actor.config
        )

    @torch.no_grad()
    def act(
        self, observation: Any, *, deterministic: bool = False
    ) -> Mapping[str, Tensor]:
        """Produce an action and provenance-safe behavior-policy metadata.

        A deterministic mode action is still sourced from the policy, but is
        deliberately excluded from PPO because it was not sampled from the
        bounded behavior distribution.
        """

        features, reference = self.encode_reference(observation)
        distribution = self.distribution(features, reference)
        if deterministic:
            residual = distribution.mode()
            log_prob = torch.full_like(distribution.log_prob(residual), torch.nan)
            policy_mask = torch.zeros_like(log_prob, dtype=torch.bool)
        else:
            residual, log_prob = distribution.rsample()
            policy_mask = torch.ones_like(log_prob, dtype=torch.bool)
        action_source = torch.full_like(log_prob, ActionSource.POLICY, dtype=torch.int8)
        return {
            "actions": reference + residual,
            "reference_actions": reference,
            "residual_actions": residual,
            "features": features,
            "log_probs": log_prob,
            "old_log_probs": log_prob,
            "action_source": action_source,
            "policy_mask": policy_mask,
        }


def initialization_report(
    base_policy: nn.Module, actor: ResidualActor, critic: TwinQCritic
) -> dict[str, int]:
    """Return auditable parameter counts for logs and run manifests."""

    return {
        "base_total": sum(value.numel() for value in base_policy.parameters()),
        "base_trainable": trainable_parameter_count(base_policy),
        "residual_actor_trainable": trainable_parameter_count(actor),
        "twin_q_trainable": trainable_parameter_count(critic),
    }
