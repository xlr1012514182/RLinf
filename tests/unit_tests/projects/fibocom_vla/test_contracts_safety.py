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
import pytest

from rlinf.projects.fibocom_vla.config import StackConfig
from rlinf.projects.fibocom_vla.contracts import ActionChunk, PolicyOutput, RobotState
from rlinf.projects.fibocom_vla.errors import SafetyViolationError, ShapeMismatchError
from rlinf.projects.fibocom_vla.safety import ActionSafetyGate, JointSafetyEnvelope


def test_default_stack_config_is_cross_field_valid() -> None:
    config = StackConfig()
    config.validate()
    assert config.robot.action_dim == config.residual_rl.action_dim


def test_contracts_normalize_float_arrays() -> None:
    state = RobotState(
        joint_positions=[0, 1],
        joint_names=("a", "b"),
    )
    chunk = ActionChunk(
        values=[[0, 1], [1, 2]],
        period_s=0.05,
        source_observation_ns=state.timestamp_ns,
    )
    assert state.joint_positions.dtype == np.float32
    assert chunk.values.dtype == np.float32
    assert chunk.horizon == 2
    assert chunk.action_dim == 2


def test_safety_gate_clips_position_and_step() -> None:
    gate = ActionSafetyGate(
        JointSafetyEnvelope(
            lower=np.array([-1.0, -1.0]),
            upper=np.array([1.0, 1.0]),
            max_step=np.array([0.1, 0.2]),
        )
    )
    safe = gate.filter(np.array([2.0, -2.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(safe, [0.1, -0.2])


def test_safety_gate_latches_nonfinite_estop() -> None:
    gate = ActionSafetyGate(
        JointSafetyEnvelope(
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            max_step=np.array([0.1]),
        )
    )
    with pytest.raises(SafetyViolationError, match="non-finite"):
        gate.filter(np.array([np.nan]), np.array([0.0]))
    with pytest.raises(SafetyViolationError, match="e-stop"):
        gate.filter(np.array([0.0]), np.array([0.0]))


def test_safety_envelope_rejects_nonfinite_step_limit() -> None:
    with pytest.raises(ShapeMismatchError, match="finite"):
        JointSafetyEnvelope(
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            max_step=np.array([np.nan]),
        )


def test_safety_gate_estops_an_already_out_of_bounds_state() -> None:
    gate = ActionSafetyGate(
        JointSafetyEnvelope(
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            max_step=np.array([0.1]),
        )
    )

    with pytest.raises(SafetyViolationError, match="outside"):
        gate.filter(np.array([0.0]), np.array([2.0]))
    assert gate.estopped


def test_action_chunk_and_policy_output_reject_nonfinite_timing() -> None:
    with pytest.raises(ShapeMismatchError, match="finite"):
        ActionChunk(
            values=np.zeros((1, 1), dtype=np.float32),
            period_s=float("nan"),
            source_observation_ns=1,
        )
    action = ActionChunk(
        values=np.zeros((1, 1), dtype=np.float32),
        period_s=0.05,
        source_observation_ns=1,
    )
    with pytest.raises(ShapeMismatchError, match="finite"):
        PolicyOutput(
            action=action,
            model_latency_ms=float("nan"),
            path="invalid",
        )


def test_action_chunk_suffix_keeps_model_space_time_alignment() -> None:
    chunk = ActionChunk(
        values=np.arange(6, dtype=np.float32).reshape(3, 2),
        model_values=np.arange(12, dtype=np.float32).reshape(3, 4),
        period_s=0.05,
        source_observation_ns=1,
        generated_ns=2,
        committed_prefix=2,
    )

    suffix = chunk.suffix(1)

    np.testing.assert_allclose(suffix.values, chunk.values[1:])
    np.testing.assert_allclose(suffix.model_values, chunk.model_values[1:])
    assert suffix.generated_ns == chunk.generated_ns
    assert suffix.committed_prefix == 1
