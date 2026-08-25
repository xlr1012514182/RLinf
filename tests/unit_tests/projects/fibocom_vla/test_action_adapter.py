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
from pathlib import Path

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import (
    ActionAdapterConfig,
    CameraConfig,
    ResidualRLConfig,
    RobotConfig,
    RTCConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.contracts import ActionChunk, Observation, RobotState
from rlinf.projects.fibocom_vla.errors import (
    ConfigurationError,
    SafetyViolationError,
    ShapeMismatchError,
)
from rlinf.projects.fibocom_vla.hardware.action_adapter import (
    PolicyRobotActionAdapter,
)


def _spec(
    mode: str,
    *,
    validated: bool = True,
    required_cameras: tuple[str, ...] = ("main", "wrist"),
) -> ActionAdapterConfig:
    return ActionAdapterConfig(
        action_mode=mode,
        policy_frame="joint",
        policy_dim=2,
        policy_names=("p0", "p1"),
        policy_units=("radian", "radian"),
        policy_state_frame="joint",
        policy_state_dim=2,
        policy_state_names=("p0", "p1"),
        required_cameras=required_cameras,
        robot_units=("degree", "degree"),
        robot_from_policy=(1, 0),
        scale=(2.0, -4.0),
        offset=(10.0, 20.0),
        validated=validated,
        calibration_id="unit-test-affine-v1" if validated else "",
    )


def _robot_config(
    mode: str,
    *,
    validated: bool = True,
    dry_run: bool = True,
    lower: tuple[float, float] = (-100.0, -100.0),
    upper: tuple[float, float] = (100.0, 100.0),
    max_step: tuple[float, float] = (100.0, 100.0),
) -> RobotConfig:
    return RobotConfig(
        action_dim=2,
        control_hz=10.0,
        joint_names=("r0", "r1"),
        joint_lower=lower,
        joint_upper=upper,
        max_step=max_step,
        dry_run=dry_run,
        action_adapter=_spec(mode, validated=validated),
    )


def _observation(
    positions: tuple[float, float] = (14.0, 16.0),
    *,
    timestamp_ns: int = 1_000,
    cameras: tuple[str, ...] = ("main", "wrist"),
) -> Observation:
    state = RobotState(
        joint_positions=np.asarray(positions, dtype=np.float32),
        joint_velocities=np.asarray([6.0, -8.0], dtype=np.float32),
        joint_names=("r0", "r1"),
        timestamp_ns=timestamp_ns - 10,
    )
    return Observation(
        state=state,
        images={name: np.zeros((2, 2, 3), dtype=np.uint8) for name in cameras},
        instruction="stack",
        timestamp_ns=timestamp_ns,
    )


def _chunk(values, *, timestamp_ns: int = 1_000, period_s: float = 0.1):
    return ActionChunk(
        values=np.asarray(values, dtype=np.float32),
        period_s=period_s,
        source_observation_ns=timestamp_ns,
        generated_ns=timestamp_ns + 1,
    )


def _identity_spec(*, validated: bool) -> ActionAdapterConfig:
    return ActionAdapterConfig(
        action_mode="absolute",
        policy_dim=2,
        policy_names=("r0", "r1"),
        policy_units=("radian", "radian"),
        policy_state_dim=2,
        policy_state_names=("r0", "r1"),
        robot_units=("radian", "radian"),
        robot_from_policy=(0, 1),
        scale=(1.0, 1.0),
        offset=(0.0, 0.0),
        validated=validated,
        calibration_id="unit-test-identity-v1" if validated else "",
    )


def test_absolute_affine_mapping_and_reverse_state_are_consistent() -> None:
    adapter = PolicyRobotActionAdapter(_robot_config("absolute"))
    observation = _observation()

    policy_state = adapter.observation_to_policy_state(observation)
    np.testing.assert_allclose(policy_state.joint_positions, [1.0, 2.0])
    np.testing.assert_allclose(policy_state.joint_velocities, [2.0, 3.0])
    assert policy_state.joint_names == ("p0", "p1")
    output = adapter.adapt_action_chunk(_chunk([[3.0, 4.0], [4.0, 5.0]]), observation)

    np.testing.assert_allclose(output.values, [[18.0, 8.0], [20.0, 4.0]])
    assert output.metadata["policy_robot_adapter"]["output_semantics"] == (
        "absolute_robot_joint_target"
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "delta_from_observation",
            [[14.4, 15.6], [14.8, 14.8]],
        ),
        (
            "integrated_delta",
            [[14.4, 15.6], [15.2, 14.4]],
        ),
        (
            "velocity",
            [[10.04, 19.96], [10.12, 19.84]],
        ),
    ],
)
def test_relative_action_modes_convert_before_robot_limits(
    mode: str, expected: list[list[float]]
) -> None:
    adapter = PolicyRobotActionAdapter(_robot_config(mode))
    observation = _observation()
    values = [[0.1, 0.2], [0.3, 0.4]]
    if mode == "velocity":
        observation = _observation(positions=(10.0, 20.0))

    output = adapter.adapt_action_chunk(_chunk(values), observation)

    np.testing.assert_allclose(output.values, expected, atol=1e-6)


def test_dry_run_may_use_unvalidated_identity_but_real_motion_may_not() -> None:
    dry_config = replace(
        _robot_config("absolute", validated=False),
        action_adapter=ActionAdapterConfig(),
    )
    adapter = PolicyRobotActionAdapter(dry_config)
    identity_observation = Observation(
        state=RobotState([0.0, 0.0], ("r0", "r1"), timestamp_ns=990),
        images={},
        instruction="stack",
        timestamp_ns=1_000,
    )
    output = adapter.adapt_action_chunk(_chunk([[0.1, -0.1]]), identity_observation)
    np.testing.assert_allclose(output.values, [[0.1, -0.1]])

    real_config = replace(dry_config, dry_run=False)
    with pytest.raises(ConfigurationError, match="validated=true"):
        real_config.validate()


def test_validated_nonidentity_mapping_is_admitted_for_policy_wrapper() -> None:
    config = _robot_config("absolute", dry_run=False)

    config.validate()
    assert PolicyRobotActionAdapter(config).config.validated


def test_validated_absolute_joint_identity_remains_supported() -> None:
    config = replace(
        _robot_config("absolute"),
        dry_run=False,
        action_adapter=_identity_spec(validated=True),
    )

    config.validate()
    assert PolicyRobotActionAdapter(config).config.validated


def test_checkpoint_native_or_end_effector_contract_fails_without_kinematics() -> None:
    placeholder = ActionAdapterConfig(
        action_mode="delta_from_observation",
        policy_frame="end_effector",
        policy_dim=2,
        policy_names=("delta_x", "gripper"),
        policy_units=("meter", "normalized"),
        policy_state_frame="checkpoint_native",
        policy_state_dim=3,
        policy_state_names=("eef_x", "eef_y", "gripper"),
        required_cameras=("main", "wrist"),
        validated=False,
    )
    config = replace(
        _robot_config("absolute", validated=False), action_adapter=placeholder
    )
    config.validate()

    with pytest.raises(ConfigurationError, match="reviewed IK"):
        PolicyRobotActionAdapter(config)


def test_required_camera_contract_is_checked_in_config_and_observation() -> None:
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
        rtc=RTCConfig(execution_horizon=1, overlap_horizon=1),
        robot=_robot_config("absolute"),
        cameras=(CameraConfig(name="main"),),
    )
    with pytest.raises(ConfigurationError, match="undeclared cameras: wrist"):
        config.validate()

    adapter = PolicyRobotActionAdapter(_robot_config("absolute"))
    with pytest.raises(ShapeMismatchError, match="required policy cameras: wrist"):
        adapter.adapt_action_chunk(
            _chunk([[1.0, 2.0]]), _observation(cameras=("main",))
        )


def test_dimension_mapping_and_numeric_contracts_fail_closed() -> None:
    duplicate_mapping = replace(_spec("absolute"), robot_from_policy=(0, 0))
    config = replace(_robot_config("absolute"), action_adapter=duplicate_mapping)
    with pytest.raises(ConfigurationError, match="indices must be unique"):
        config.validate()

    nan_rate = replace(_robot_config("absolute"), control_hz=float("nan"))
    with pytest.raises(ConfigurationError, match="control_hz"):
        nan_rate.validate()

    adapter = PolicyRobotActionAdapter(_robot_config("absolute"))
    observation = _observation()
    with pytest.raises(ShapeMismatchError, match="policy action dim"):
        adapter.adapt_action_chunk(_chunk([[1.0, 2.0, 3.0]]), observation)
    with pytest.raises(ShapeMismatchError, match="finite and positive"):
        adapter.adapt_action_chunk(
            _chunk([[1.0, 2.0]], period_s=float("nan")), observation
        )
    with pytest.raises(ShapeMismatchError, match="control period"):
        adapter.adapt_action_chunk(_chunk([[1.0, 2.0]], period_s=0.2), observation)
    mutable = _chunk([[1.0, 2.0]])
    mutable.values[0, 0] = np.nan
    with pytest.raises(ShapeMismatchError, match="policy action chunk"):
        adapter.adapt_action_chunk(mutable, observation)


def test_timestamp_and_joint_order_contracts_fail_closed() -> None:
    adapter = PolicyRobotActionAdapter(_robot_config("absolute"))
    observation = _observation()
    with pytest.raises(ShapeMismatchError, match="anchored"):
        adapter.adapt_action_chunk(
            _chunk([[1.0, 2.0]], timestamp_ns=2_000), observation
        )
    with pytest.raises(ShapeMismatchError, match="predate"):
        replace(_chunk([[1.0, 2.0]]), generated_ns=999)
    wrong_order = replace(
        observation,
        state=RobotState([14.0, 16.0], ("r1", "r0"), timestamp_ns=990),
    )
    with pytest.raises(ShapeMismatchError, match="calibrated robot order"):
        adapter.adapt_action_chunk(_chunk([[1.0, 2.0]]), wrong_order)


def test_semantic_conversion_precedes_position_and_step_limit_checks() -> None:
    position_limited = PolicyRobotActionAdapter(
        _robot_config(
            "delta_from_observation",
            lower=(-10.0, -10.0),
            upper=(15.0, 20.0),
        )
    )
    with pytest.raises(SafetyViolationError, match="joint limits"):
        position_limited.adapt_action_chunk(_chunk([[0.0, 1.0]]), _observation())

    step_limited = PolicyRobotActionAdapter(
        _robot_config(
            "integrated_delta",
            max_step=(0.1, 0.1),
        )
    )
    with pytest.raises(SafetyViolationError, match="max_step"):
        step_limited.adapt_action_chunk(_chunk([[0.1, 0.1]]), _observation())


def test_checked_in_action_adapters_cover_schema_keys() -> None:
    project_root = Path(__file__).resolve().parents[4]
    example_root = project_root / "examples" / "embodiment" / "fibocom_vla"
    schema = json.loads(
        (example_root / "schema" / "action_adapter.schema.json").read_text(
            encoding="utf-8"
        )
    )
    required = set(schema["required"])
    allowed = set(schema["properties"])

    for config_path in sorted((example_root / "config").glob("*.json")):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        adapter = payload["robot"]["action_adapter"]
        assert required <= set(adapter), config_path.name
        assert set(adapter) <= allowed, config_path.name
        StackConfig.from_json(config_path)
