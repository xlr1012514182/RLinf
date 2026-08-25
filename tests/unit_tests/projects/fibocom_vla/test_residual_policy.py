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
from rlinf.projects.fibocom_vla.contracts import (
    ActionChunk,
    Observation,
    PolicyOutput,
    RobotState,
)
from rlinf.projects.fibocom_vla.residual_policy import (
    FrozenReferenceOutput,
    ResidualChunkPolicy,
)
from rlinf.projects.fibocom_vla.rl.torch_modules import ResidualActor


def _config() -> ResidualRLConfig:
    return ResidualRLConfig(
        feature_dim=4,
        action_horizon=2,
        action_dim=1,
        actor_hidden_dim=4,
        actor_bottleneck_dim=2,
        target_actor_parameters=1,
        parameter_tolerance=10_000,
    )


def _observation() -> Observation:
    return Observation(
        state=RobotState(np.zeros(1, dtype=np.float32), ("joint",)),
        images={"main": np.zeros((4, 4, 3), dtype=np.uint8)},
        instruction="move",
    )


class _FrozenPolicy:
    def __init__(self, config: ResidualRLConfig) -> None:
        self.config = config

    def predict_with_features(self, observation: Observation) -> FrozenReferenceOutput:
        return FrozenReferenceOutput(
            output=PolicyOutput(
                action=ActionChunk(
                    values=np.zeros(
                        (self.config.action_horizon, self.config.action_dim),
                        dtype=np.float32,
                    ),
                    period_s=0.05,
                    source_observation_ns=observation.timestamp_ns,
                ),
                model_latency_ms=0.1,
                path="frozen",
            ),
            features=np.zeros(self.config.feature_dim, dtype=np.float32),
        )


def test_deterministic_residual_policy_is_not_ppo_eligible() -> None:
    config = _config()
    actor = ResidualActor(config)
    policy = ResidualChunkPolicy(
        _FrozenPolicy(config), actor, checkpoint_sha256="0" * 64
    )

    behavior = policy.predict_behavior(_observation())

    assert behavior.behavior_log_prob is None
    assert not behavior.stochastic_policy_sample_unchanged
    assert behavior.output.diagnostics["ppo_eligible"] is False
    np.testing.assert_allclose(
        behavior.output.action.values,
        behavior.reference_actions + behavior.residual_actions,
    )


def test_stochastic_residual_policy_records_actual_behavior_density() -> None:
    torch.manual_seed(7)
    config = _config()
    policy = ResidualChunkPolicy(
        _FrozenPolicy(config),
        ResidualActor(config),
        checkpoint_sha256="1" * 64,
        stochastic=True,
    )

    behavior = policy.predict_behavior(_observation())

    assert behavior.behavior_log_prob is not None
    assert np.isfinite(behavior.behavior_log_prob)
    assert behavior.stochastic_policy_sample_unchanged
    assert behavior.output.diagnostics["ppo_eligible"] is True
    assert np.max(np.abs(behavior.residual_actions)) <= config.max_residual
