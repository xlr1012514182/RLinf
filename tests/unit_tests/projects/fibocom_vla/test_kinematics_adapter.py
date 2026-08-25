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

import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import ActionAdapterConfig, RobotConfig
from rlinf.projects.fibocom_vla.contracts import ActionChunk, Observation, RobotState
from rlinf.projects.fibocom_vla.errors import (
    ConfigurationError,
    SafetyViolationError,
    ShapeMismatchError,
)
from rlinf.projects.fibocom_vla.hardware.kinematics import (
    create_libero_cartesian_adapter,
)


def _install_engine_factory(monkeypatch, name: str, factory) -> str:
    module = types.ModuleType(name)
    module.create_engine = factory
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:create_engine"


def _robot_config(
    kinematics_factory: object,
    *,
    max_step: tuple[float, float] = (0.5, 0.5),
) -> RobotConfig:
    return RobotConfig(
        action_dim=2,
        control_hz=10.0,
        joint_names=("joint_0", "joint_1"),
        joint_lower=(-1.0, -1.0),
        joint_upper=(1.0, 1.0),
        max_step=max_step,
        dry_run=True,
        action_adapter=ActionAdapterConfig(
            implementation=(
                "rlinf.projects.fibocom_vla.hardware.kinematics:"
                "create_libero_cartesian_adapter"
            ),
            action_mode="delta_from_observation",
            coordinate_modes=(
                "delta_from_observation",
                "delta_from_observation",
                "absolute",
            ),
            policy_frame="end_effector",
            policy_dim=3,
            policy_names=("delta_x", "delta_y", "gripper"),
            policy_units=("meter", "meter", "normalized"),
            policy_state_frame="checkpoint_native",
            policy_state_dim=4,
            policy_state_names=("eef_x", "eef_y", "eef_z", "gripper"),
            required_cameras=("main", "wrist"),
            robot_units=("radian", "radian"),
            validated=False,
        ),
        options={"kinematics_factory": kinematics_factory},
    )


def _observation(*, cameras: tuple[str, ...] = ("main", "wrist")) -> Observation:
    return Observation(
        state=RobotState(
            joint_positions=np.asarray([0.0, 0.0], dtype=np.float32),
            joint_names=("joint_0", "joint_1"),
            timestamp_ns=990,
            fault="latched-fault-for-provenance",
        ),
        images={name: np.zeros((2, 3, 3), dtype=np.uint8) for name in cameras},
        instruction="stack the blocks",
        timestamp_ns=1_000,
        frame_id=4,
    )


def _chunk(values=None) -> ActionChunk:
    if values is None:
        values = [[0.1, 0.2, 0.8], [0.2, 0.4, 0.1]]
    return ActionChunk(
        values=np.asarray(values, dtype=np.float32),
        period_s=0.1,
        source_observation_ns=1_000,
        generated_ns=1_010,
        model_values=np.arange(10, dtype=np.float32).reshape(2, 5),
        committed_prefix=1,
        metadata={"checkpoint": "pi05"},
    )


class _FakeKinematicsEngine:
    def encode_policy_state(self, observation: Observation):
        del observation
        return np.asarray([0.1, 0.2, 0.3, 0.8], dtype=np.float32)

    def policy_chunk_to_robot_targets(
        self,
        observation: Observation,
        policy_values,
        period_s: float,
        action_adapter_config: ActionAdapterConfig,
    ):
        assert observation.timestamp_ns == 1_000
        assert period_s == 0.1
        assert action_adapter_config.policy_frame == "end_effector"
        return np.asarray(policy_values, dtype=np.float32)[:, :2]


def test_fake_engine_adapts_state_and_absolute_targets_with_provenance(
    monkeypatch,
) -> None:
    engine = _FakeKinematicsEngine()
    factory_name = _install_engine_factory(
        monkeypatch, "fibocom_test_kinematics_ok", lambda config: engine
    )
    adapter = create_libero_cartesian_adapter(_robot_config(factory_name))
    observation = _observation()

    state = adapter.observation_to_policy_state(observation)
    output = adapter.adapt_action_chunk(_chunk(), observation)

    np.testing.assert_allclose(state.joint_positions, [0.1, 0.2, 0.3, 0.8])
    assert state.joint_names == ("eef_x", "eef_y", "eef_z", "gripper")
    assert state.timestamp_ns == observation.state.timestamp_ns
    assert state.fault == observation.state.fault
    np.testing.assert_allclose(output.values, [[0.1, 0.2], [0.2, 0.4]])
    assert output.horizon == 2
    assert output.period_s == 0.1
    assert output.source_observation_ns == 1_000
    assert output.generated_ns == 1_010
    assert output.committed_prefix == 1
    assert output.model_values is None
    assert output.metadata["checkpoint"] == "pi05"
    evidence = output.metadata["libero_cartesian_kinematics"]
    assert evidence["engine_factory"] == factory_name
    assert evidence["output_semantics"] == "absolute_robot_joint_target"
    assert evidence["rtc_native_safe"] is False
    assert adapter.rtc_native_safe is False


@pytest.mark.parametrize(
    "factory_value",
    [None, "", "REPLACE_WITH_MODULE:create_engine", "module:function"],
)
def test_missing_or_placeholder_engine_factory_fails_closed(factory_value) -> None:
    with pytest.raises(ConfigurationError, match="kinematics_factory|placeholder"):
        create_libero_cartesian_adapter(_robot_config(factory_value))


def test_engine_factory_requires_complete_interface(monkeypatch) -> None:
    factory_name = _install_engine_factory(
        monkeypatch,
        "fibocom_test_kinematics_incomplete",
        lambda config: object(),
    )

    with pytest.raises(ConfigurationError, match="missing callable"):
        create_libero_cartesian_adapter(_robot_config(factory_name))


def test_policy_state_schema_and_camera_contract_fail_closed(monkeypatch) -> None:
    class WrongStateEngine(_FakeKinematicsEngine):
        def encode_policy_state(self, observation: Observation):
            del observation
            return np.asarray([0.1, 0.2], dtype=np.float32)

    factory_name = _install_engine_factory(
        monkeypatch,
        "fibocom_test_kinematics_wrong_state",
        lambda config: WrongStateEngine(),
    )
    adapter = create_libero_cartesian_adapter(_robot_config(factory_name))

    with pytest.raises(ShapeMismatchError, match=r"shape \(4,\)"):
        adapter.observation_to_policy_state(_observation())
    with pytest.raises(ShapeMismatchError, match="required policy cameras: wrist"):
        adapter.observation_to_policy_state(_observation(cameras=("main",)))


@pytest.mark.parametrize("failure", ["limit", "step", "shape", "nan"])
def test_engine_targets_are_revalidated_without_clipping(
    monkeypatch, failure: str
) -> None:
    class InvalidTargetEngine(_FakeKinematicsEngine):
        def policy_chunk_to_robot_targets(
            self, observation, policy_values, period_s, action_adapter_config
        ):
            del observation, policy_values, period_s, action_adapter_config
            if failure == "limit":
                return [[1.1, 0.0], [1.1, 0.0]]
            if failure == "step":
                return [[0.6, 0.0], [0.6, 0.0]]
            if failure == "shape":
                return [[0.0], [0.0]]
            return [[0.0, np.nan], [0.0, 0.0]]

    factory_name = _install_engine_factory(
        monkeypatch,
        f"fibocom_test_kinematics_invalid_{failure}",
        lambda config: InvalidTargetEngine(),
    )
    adapter = create_libero_cartesian_adapter(_robot_config(factory_name))

    if failure in {"limit", "step"}:
        expected_exception = SafetyViolationError
    else:
        expected_exception = ShapeMismatchError
    with pytest.raises(expected_exception):
        adapter.adapt_action_chunk(_chunk(), _observation())


def test_chunk_dimension_period_and_timestamp_are_checked_before_ik(
    monkeypatch,
) -> None:
    factory_name = _install_engine_factory(
        monkeypatch,
        "fibocom_test_kinematics_contract",
        lambda config: _FakeKinematicsEngine(),
    )
    adapter = create_libero_cartesian_adapter(_robot_config(factory_name))
    observation = _observation()

    with pytest.raises(ShapeMismatchError, match=r"shape \[T, 3\]"):
        adapter.adapt_action_chunk(_chunk([[0.1, 0.2], [0.2, 0.3]]), observation)
    with pytest.raises(ShapeMismatchError, match="control period"):
        adapter.adapt_action_chunk(replace(_chunk(), period_s=0.2), observation)
    with pytest.raises(ShapeMismatchError, match="anchored"):
        adapter.adapt_action_chunk(
            replace(_chunk(), source_observation_ns=2_000, generated_ns=2_010),
            observation,
        )


def test_robotstate_engine_output_must_preserve_names_and_timestamp(
    monkeypatch,
) -> None:
    class WrongRobotStateEngine(_FakeKinematicsEngine):
        def encode_policy_state(self, observation: Observation):
            return RobotState(
                joint_positions=np.zeros(4, dtype=np.float32),
                joint_names=("wrong_0", "wrong_1", "wrong_2", "wrong_3"),
                timestamp_ns=observation.state.timestamp_ns + 1,
            )

    factory_name = _install_engine_factory(
        monkeypatch,
        "fibocom_test_kinematics_robot_state",
        lambda config: WrongRobotStateEngine(),
    )
    adapter = create_libero_cartesian_adapter(_robot_config(factory_name))

    with pytest.raises(ShapeMismatchError, match="names"):
        adapter.observation_to_policy_state(_observation())
