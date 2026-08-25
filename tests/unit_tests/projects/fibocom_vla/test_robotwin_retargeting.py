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
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import (
    ActionAdapterConfig,
    RobotConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.contracts import ActionChunk, Observation, RobotState
from rlinf.projects.fibocom_vla.errors import (
    ConfigurationError,
    SafetyViolationError,
    ShapeMismatchError,
)
from rlinf.projects.fibocom_vla.hardware.policy_adapter import (
    PolicyRobotAdapterPolicy,
    load_policy_robot_adapter,
)
from rlinf.projects.fibocom_vla.hardware.robotwin_retargeting import (
    ROBOTWIN_PI05_CAMERAS,
    ROBOTWIN_PI05_JOINTS,
    create_robotwin_pi05_h50_retargeting_adapter,
)

_SOURCE_UNITS = (
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "normalized",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "normalized",
)
_IMPLEMENTATION = (
    "rlinf.projects.fibocom_vla.hardware.robotwin_retargeting:"
    "create_robotwin_pi05_h50_retargeting_adapter"
)


def _install_engine_factory(monkeypatch, name: str, factory) -> str:
    module = types.ModuleType(name)
    module.create_engine = factory
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:create_engine"


def _adapter_config(*, validated: bool = False) -> ActionAdapterConfig:
    return ActionAdapterConfig(
        implementation=_IMPLEMENTATION,
        action_mode="absolute",
        coordinate_modes=("absolute",) * 14,
        policy_frame="joint",
        policy_dim=14,
        policy_names=ROBOTWIN_PI05_JOINTS,
        policy_units=_SOURCE_UNITS,
        policy_state_frame="joint",
        policy_state_dim=14,
        policy_state_names=ROBOTWIN_PI05_JOINTS,
        required_cameras=ROBOTWIN_PI05_CAMERAS,
        robot_units=("degree",) * 5 + ("percent",),
        validated=validated,
        calibration_id=(
            "measured-so101-retargeting-v1"
            if validated
            else "LOCKED_DRY_RUN_RETARGETING"
        ),
    )


def _robot_config(
    engine_factory: object,
    *,
    backend: str = "so101",
    dry_run: bool = True,
    validated: bool = False,
    max_step: float = 0.5,
) -> RobotConfig:
    if backend == "so101":
        dim = 6
        names = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
        lower = (-2.0,) * dim
        upper = (2.0,) * dim
        adapter_config = _adapter_config(validated=validated)
    else:
        dim = 7
        names = (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        )
        lower = (-2.0,) * 6 + (0.0,)
        upper = (2.0,) * 6 + (1.0,)
        adapter_config = replace(
            _adapter_config(validated=validated),
            robot_units=("degree",) * 6 + ("binary",),
            calibration_id=(
                "measured-nova-retargeting-v1"
                if validated
                else "LOCKED_DRY_RUN_RETARGETING"
            ),
        )
    return RobotConfig(
        backend=backend,
        action_dim=dim,
        control_hz=20.0,
        maximum_state_camera_skew_ms=75.0,
        joint_names=names,
        joint_lower=lower,
        joint_upper=upper,
        max_step=(max_step,) * dim,
        dry_run=dry_run,
        action_adapter=adapter_config,
        options={"robotwin_retargeting_factory": engine_factory},
    )


def _observation(
    config: RobotConfig,
    *,
    cameras: tuple[str, ...] = ROBOTWIN_PI05_CAMERAS,
    metadata: dict | None = None,
    state_timestamp_ns: int = 9_990_000_000,
    observation_timestamp_ns: int = 10_000_000_000,
    fault: str | None = None,
) -> Observation:
    if metadata is None:
        metadata = {
            "camera_timestamps_ns": {
                "cam_high": 9_998_000_000,
                "cam_left_wrist": 9_999_000_000,
                "cam_right_wrist": observation_timestamp_ns,
            }
        }
    return Observation(
        state=RobotState(
            joint_positions=np.zeros(config.action_dim, dtype=np.float32),
            joint_names=tuple(config.joint_names),
            timestamp_ns=state_timestamp_ns,
            fault=fault,
        ),
        images={name: np.zeros((8, 8, 3), dtype=np.uint8) for name in cameras},
        instruction="stack the blocks",
        timestamp_ns=observation_timestamp_ns,
        frame_id=4,
        metadata=metadata,
    )


def _chunk(*, horizon: int = 50, action_dim: int = 14) -> ActionChunk:
    return ActionChunk(
        values=np.zeros((horizon, action_dim), dtype=np.float32),
        period_s=0.05,
        source_observation_ns=10_000_000_000,
        generated_ns=10_000_100_000,
        model_values=np.zeros((horizon, 32), dtype=np.float32),
        committed_prefix=min(3, horizon),
        metadata={"checkpoint": "pi05_aloha_robotwin"},
    )


class _Engine:
    def __init__(self, robot_dim: int) -> None:
        self.robot_dim = robot_dim
        self.target_mode = "valid"
        self.seen_policy_values = None

    def encode_policy_state(self, observation: Observation):
        assert set(ROBOTWIN_PI05_CAMERAS) <= set(observation.images)
        return np.linspace(-0.2, 0.2, 14, dtype=np.float32)

    def policy_chunk_to_robot_targets(
        self,
        observation: Observation,
        policy_values,
        period_s: float,
        config: ActionAdapterConfig,
    ):
        assert observation.timestamp_ns == 10_000_000_000
        assert period_s == 0.05
        assert config.policy_dim == 14
        self.seen_policy_values = policy_values
        targets = np.zeros((50, self.robot_dim), dtype=np.float32)
        if self.target_mode == "limit":
            targets[:, 0] = 3.0
        elif self.target_mode == "step":
            targets[:, 0] = 0.75
        elif self.target_mode == "shape":
            return targets[:-1]
        return targets


@pytest.mark.parametrize("backend", ["so101", "dobot_nova"])
def test_engine_boundary_adapts_h50_state_and_absolute_targets(
    monkeypatch, backend: str
) -> None:
    dim = 6 if backend == "so101" else 7
    engine = _Engine(dim)
    factory_name = _install_engine_factory(
        monkeypatch,
        f"fibocom_robotwin_{backend}_engine",
        lambda config: engine,
    )
    config = _robot_config(factory_name, backend=backend)

    adapter = load_policy_robot_adapter(config)
    observation = _observation(config)
    state = adapter.observation_to_policy_state(observation)
    output = adapter.adapt_action_chunk(_chunk(), observation)

    assert state.joint_names == ROBOTWIN_PI05_JOINTS
    assert state.joint_positions.shape == (14,)
    assert state.timestamp_ns == observation.state.timestamp_ns
    assert output.values.shape == (50, dim)
    assert output.committed_prefix == 3
    assert output.model_values is None
    assert output.metadata["robotwin_pi05_h50_retargeting"] == {
        "engine_factory": factory_name,
        "source_contract": "robotwin_dual_aloha_absolute_h50_a14",
        "output_semantics": "absolute_robot_joint_target",
        "calibration_validated": False,
        "calibration_id": "LOCKED_DRY_RUN_RETARGETING",
        "rtc_native_safe": False,
    }
    assert adapter.rtc_native_safe is False


def test_policy_wrapper_keeps_native_rtc_closed(monkeypatch) -> None:
    engine = _Engine(6)
    factory_name = _install_engine_factory(
        monkeypatch, "fibocom_robotwin_rtc_closed", lambda config: engine
    )
    config = _robot_config(factory_name)

    class NativePolicy:
        rtc_native_supported = True

        def predict(self, observation):
            del observation
            raise AssertionError("not exercised")

        def predict_with_rtc(self, observation, conditioning):
            del observation, conditioning
            raise AssertionError("must remain unreachable")

    wrapped = PolicyRobotAdapterPolicy(NativePolicy(), config)
    assert wrapped.rtc_native_supported is False


@pytest.mark.parametrize(
    "factory_value",
    [
        None,
        "",
        "module:function",
        "LOCKED_PLACEHOLDER_REPLACE_WITH_ENGINE:create_engine",
        "missing_separator",
    ],
)
def test_factory_rejects_missing_or_placeholder_engine(factory_value) -> None:
    with pytest.raises(ConfigurationError):
        create_robotwin_pi05_h50_retargeting_adapter(_robot_config(factory_value))


def test_factory_requires_both_engine_methods(monkeypatch) -> None:
    factory_name = _install_engine_factory(
        monkeypatch,
        "fibocom_robotwin_incomplete_engine",
        lambda config: object(),
    )
    with pytest.raises(ConfigurationError, match="missing callable"):
        create_robotwin_pi05_h50_retargeting_adapter(_robot_config(factory_name))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_dim", 13, "length 13"),
        ("policy_state_dim", 13, "length 13"),
        ("policy_names", tuple(f"p{i}" for i in range(14)), "policy_names"),
        (
            "required_cameras",
            ("cam_high", "cam_right_wrist", "cam_left_wrist"),
            "source order",
        ),
        ("coordinate_modes", ("absolute",) * 13 + ("velocity",), "absolute"),
        ("robot_from_policy", tuple(range(6)), "affine"),
    ],
)
def test_static_checkpoint_contract_is_source_locked(
    monkeypatch, field: str, value, message: str
) -> None:
    engine = _Engine(6)
    factory_name = _install_engine_factory(
        monkeypatch,
        f"fibocom_robotwin_bad_{field}",
        lambda config: engine,
    )
    config = _robot_config(factory_name)
    config = replace(
        config, action_adapter=replace(config.action_adapter, **{field: value})
    )
    with pytest.raises(ConfigurationError, match=message):
        create_robotwin_pi05_h50_retargeting_adapter(config)


def test_observation_requires_all_cameras_and_timestamp_evidence(monkeypatch) -> None:
    engine = _Engine(6)
    factory_name = _install_engine_factory(
        monkeypatch, "fibocom_robotwin_timestamp_engine", lambda config: engine
    )
    config = _robot_config(factory_name)
    adapter = create_robotwin_pi05_h50_retargeting_adapter(config)

    with pytest.raises(ShapeMismatchError, match="missing required"):
        adapter.observation_to_policy_state(
            _observation(config, cameras=("cam_high", "cam_left_wrist"))
        )
    with pytest.raises(ShapeMismatchError, match="camera_timestamps_ns"):
        adapter.observation_to_policy_state(_observation(config, metadata={}))
    wrong_latest = {
        "camera_timestamps_ns": dict.fromkeys(ROBOTWIN_PI05_CAMERAS, 9_999_000_000)
    }
    with pytest.raises(ShapeMismatchError, match="latest required camera"):
        adapter.observation_to_policy_state(_observation(config, metadata=wrong_latest))
    excessive_skew = {
        "camera_timestamps_ns": {
            "cam_high": 9_000_000_000,
            "cam_left_wrist": 9_999_000_000,
            "cam_right_wrist": 10_000_000_000,
        }
    }
    with pytest.raises(ShapeMismatchError, match="exceed"):
        adapter.observation_to_policy_state(
            _observation(config, metadata=excessive_skew)
        )


def test_fault_and_out_of_limit_observations_are_rejected(monkeypatch) -> None:
    engine = _Engine(6)
    factory_name = _install_engine_factory(
        monkeypatch, "fibocom_robotwin_state_safety", lambda config: engine
    )
    config = _robot_config(factory_name)
    adapter = create_robotwin_pi05_h50_retargeting_adapter(config)

    with pytest.raises(SafetyViolationError, match="fault"):
        adapter.observation_to_policy_state(_observation(config, fault="estop"))
    observation = _observation(config)
    bad_state = replace(
        observation.state,
        joint_positions=np.asarray([3.0, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    with pytest.raises(SafetyViolationError, match="joint limits"):
        adapter.observation_to_policy_state(replace(observation, state=bad_state))


def test_chunk_requires_h50_a14_period_and_matching_provenance(monkeypatch) -> None:
    engine = _Engine(6)
    factory_name = _install_engine_factory(
        monkeypatch, "fibocom_robotwin_chunk_contract", lambda config: engine
    )
    config = _robot_config(factory_name)
    adapter = create_robotwin_pi05_h50_retargeting_adapter(config)
    observation = _observation(config)

    with pytest.raises(ShapeMismatchError, match="horizon"):
        adapter.adapt_action_chunk(_chunk(horizon=49), observation)
    with pytest.raises(ShapeMismatchError, match=r"\[50, 14\]"):
        adapter.adapt_action_chunk(_chunk(action_dim=13), observation)
    with pytest.raises(ShapeMismatchError, match="control period"):
        adapter.adapt_action_chunk(replace(_chunk(), period_s=0.1), observation)
    with pytest.raises(ShapeMismatchError, match="anchored"):
        adapter.adapt_action_chunk(
            replace(
                _chunk(),
                source_observation_ns=10_001_000_000,
                generated_ns=10_001_100_000,
            ),
            observation,
        )


@pytest.mark.parametrize(
    ("target_mode", "exception", "message"),
    [
        ("shape", ShapeMismatchError, "shape"),
        ("limit", SafetyViolationError, "joint limits"),
        ("step", SafetyViolationError, "max_step"),
    ],
)
def test_engine_targets_are_never_reshaped_clipped_or_rate_limited(
    monkeypatch, target_mode: str, exception, message: str
) -> None:
    engine = _Engine(6)
    engine.target_mode = target_mode
    factory_name = _install_engine_factory(
        monkeypatch,
        f"fibocom_robotwin_bad_target_{target_mode}",
        lambda config: engine,
    )
    config = _robot_config(factory_name)
    adapter = create_robotwin_pi05_h50_retargeting_adapter(config)

    with pytest.raises(exception, match=message):
        adapter.adapt_action_chunk(_chunk(), _observation(config))


def test_real_motion_requires_matching_engine_validation_evidence(monkeypatch) -> None:
    engine = _Engine(6)
    factory_name = _install_engine_factory(
        monkeypatch, "fibocom_robotwin_real_gate", lambda config: engine
    )
    config = _robot_config(factory_name, dry_run=False, validated=True)
    adapter = create_robotwin_pi05_h50_retargeting_adapter(config)
    observation = _observation(config)

    with pytest.raises(PermissionError, match="validated=True"):
        adapter.adapt_action_chunk(_chunk(), observation)
    engine.validated = True
    engine.calibration_id = "different-calibration"
    with pytest.raises(PermissionError, match="calibration_id"):
        adapter.adapt_action_chunk(_chunk(), observation)
    engine.calibration_id = config.action_adapter.calibration_id
    output = adapter.adapt_action_chunk(_chunk(), observation)
    assert output.values.shape == (50, 6)


def test_real_motion_rejects_placeholder_calibration_even_if_engine_matches(
    monkeypatch,
) -> None:
    engine = _Engine(6)
    engine.validated = True
    engine.calibration_id = "LOCKED_PLACEHOLDER_CALIBRATION"
    factory_name = _install_engine_factory(
        monkeypatch, "fibocom_robotwin_calibration_gate", lambda config: engine
    )
    config = _robot_config(factory_name, dry_run=False, validated=True)
    config = replace(
        config,
        action_adapter=replace(
            config.action_adapter,
            calibration_id=engine.calibration_id,
        ),
    )
    adapter = create_robotwin_pi05_h50_retargeting_adapter(config)

    with pytest.raises(PermissionError, match="non-placeholder"):
        adapter.adapt_action_chunk(_chunk(), _observation(config))


def test_locked_so101_and_nova_examples_remain_dry_run_h50_placeholders() -> None:
    project_root = Path(__file__).resolve().parents[4]
    config_root = project_root / "examples" / "embodiment" / "fibocom_vla" / "config"
    names = (
        "robotwin_pi05_h50_so101_6d_retargeting_locked.json",
        "robotwin_pi05_h50_nova_7d_retargeting_locked.json",
    )

    for name in names:
        path = config_root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = StackConfig.from_json(path)
        assert loaded.residual_rl.action_horizon == 50
        assert loaded.residual_rl.action_dim == 14
        assert loaded.rtc.enabled is False
        assert loaded.robot.dry_run is True
        assert loaded.robot.action_adapter.validated is False
        assert tuple(camera.name for camera in loaded.cameras) == ROBOTWIN_PI05_CAMERAS
        assert (
            "LOCKED_PLACEHOLDER"
            in payload["robot"]["options"]["robotwin_retargeting_factory"]
        )
        with pytest.raises(ConfigurationError, match="locked placeholder"):
            load_policy_robot_adapter(loaded.robot)
