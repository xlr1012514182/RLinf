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

"""Inference-time composition of frozen π0.5 and the residual actor."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from .config import ResidualRLConfig
from .contracts import ActionChunk, Observation, PolicyOutput
from .errors import ConfigurationError, ShapeMismatchError
from .rl.checkpoint import load_residual_checkpoint
from .rl.torch_modules import BoundedResidualDistribution, ResidualActor
from .rl.trajectory import ActionSource


@dataclass(frozen=True)
class FrozenReferenceOutput:
    """One frozen-base action and its same-forward pooled prefix feature."""

    output: PolicyOutput
    features: NDArray[np.float32]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if features.ndim != 1 or not np.all(np.isfinite(features)):
            raise ShapeMismatchError("frozen-base features must be finite [F]")
        object.__setattr__(self, "features", features)


class FrozenReferencePolicy(Protocol):
    """Policy that exposes reference actions and frozen prefix features."""

    def predict_with_features(self, observation: Observation) -> FrozenReferenceOutput:
        """Return a same-forward reference chunk and pooled feature."""


@dataclass(frozen=True)
class ResidualBehaviorOutput:
    """Execution output plus exact behavior-policy rollout fields."""

    output: PolicyOutput
    features: NDArray[np.float32]
    reference_actions: NDArray[np.float32]
    residual_actions: NDArray[np.float32]
    action_source: ActionSource
    behavior_log_prob: float | None
    stochastic_policy_sample_unchanged: bool
    checkpoint_sha256: str


class ResidualChunkPolicy:
    """Apply a checkpointed bounded residual around a frozen base chunk.

    Deployment defaults to the deterministic distribution mode. Rollout
    collection must opt into stochastic sampling; only that unmodified sample
    carries a behavior log-probability suitable for PPO.
    """

    def __init__(
        self,
        base_policy: FrozenReferencePolicy,
        actor: ResidualActor,
        *,
        checkpoint_sha256: str,
        stochastic: bool = False,
    ) -> None:
        if len(checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        self.base_policy = base_policy
        self.actor = actor.eval()
        self.config = actor.config
        self.checkpoint_sha256 = checkpoint_sha256
        self.stochastic = bool(stochastic)

    @property
    def device(self) -> torch.device:
        """Device that hosts the external residual actor."""

        return next(self.actor.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        base_policy: FrozenReferencePolicy,
        config: ResidualRLConfig,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        stochastic: bool = False,
    ) -> "ResidualChunkPolicy":
        """Load only the actor after checksum and exact-config validation."""

        payload, digest = load_residual_checkpoint(
            checkpoint, map_location=device, require_sidecar=True
        )
        trainer_state = payload["trainer_state"]
        checkpoint_config = trainer_state.get("config")
        expected_config = asdict(config)
        if checkpoint_config != expected_config:
            raise ConfigurationError(
                "residual checkpoint config differs from the active config"
            )
        actor_state = trainer_state.get("actor")
        if not isinstance(actor_state, dict):
            raise ValueError("checkpoint trainer_state is missing actor weights")
        actor = ResidualActor(config).to(device)
        actor.load_state_dict(actor_state, strict=True)
        return cls(
            base_policy,
            actor,
            checkpoint_sha256=digest,
            stochastic=stochastic,
        )

    def predict_behavior(self, observation: Observation) -> ResidualBehaviorOutput:
        """Return an adjusted chunk and provenance-safe collection fields."""

        start_ns = time.perf_counter_ns()
        frozen = self.base_policy.predict_with_features(observation)
        reference = frozen.output.action.values
        expected_action_shape = (
            self.config.action_horizon,
            self.config.action_dim,
        )
        if reference.shape != expected_action_shape:
            raise ShapeMismatchError(
                f"reference action must be {expected_action_shape}, got {reference.shape}"
            )
        if frozen.features.shape != (self.config.feature_dim,):
            raise ShapeMismatchError(
                "frozen feature shape does not match residual_rl.feature_dim: "
                f"{frozen.features.shape} != {(self.config.feature_dim,)}"
            )
        feature_tensor = torch.as_tensor(frozen.features[None, :], device=self.device)
        reference_tensor = torch.as_tensor(reference[None, :, :], device=self.device)
        with torch.no_grad():
            location, log_std = self.actor(feature_tensor, reference_tensor)
            distribution = BoundedResidualDistribution(
                location,
                log_std,
                reference_tensor,
                self.config,
            )
            if self.stochastic:
                residual_tensor, log_prob_tensor = distribution.rsample()
                behavior_log_prob = float(log_prob_tensor.item())
            else:
                residual_tensor = distribution.mode()
                behavior_log_prob = None
            adjusted_tensor = reference_tensor + residual_tensor
        residual = residual_tensor[0].float().cpu().numpy()
        adjusted = adjusted_tensor[0].float().cpu().numpy()
        total_latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        output = PolicyOutput(
            action=ActionChunk(
                values=adjusted,
                period_s=frozen.output.action.period_s,
                source_observation_ns=frozen.output.action.source_observation_ns,
                metadata={
                    **frozen.output.action.metadata,
                    "residual_actor": True,
                    "residual_checkpoint_sha256": self.checkpoint_sha256,
                    "residual_stochastic": self.stochastic,
                },
            ),
            model_latency_ms=total_latency_ms,
            path=f"{frozen.output.path}+bounded_residual_actor",
            accepted_prefix=frozen.output.accepted_prefix,
            diagnostics={
                **frozen.output.diagnostics,
                "residual_actor_parameters": self.actor.parameter_count,
                "residual_checkpoint_sha256": self.checkpoint_sha256,
                "residual_stochastic": self.stochastic,
                "ppo_eligible": self.stochastic,
                "base_model_latency_ms": frozen.output.model_latency_ms,
            },
        )
        return ResidualBehaviorOutput(
            output=output,
            features=frozen.features.copy(),
            reference_actions=reference.copy(),
            residual_actions=residual,
            action_source=ActionSource.POLICY,
            behavior_log_prob=behavior_log_prob,
            stochastic_policy_sample_unchanged=self.stochastic,
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def predict(self, observation: Observation) -> PolicyOutput:
        """Predict one deterministic or explicitly stochastic residual chunk."""

        return self.predict_behavior(observation).output
