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

import argparse
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.cli import _run
from rlinf.projects.fibocom_vla.config import (
    ActionAdapterConfig,
    CameraConfig,
    ResidualRLConfig,
    RobotConfig,
    RTCConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.contracts import RobotState
from rlinf.projects.fibocom_vla.errors import (
    ConfigurationError,
    HardwareNotReadyError,
)
from rlinf.projects.fibocom_vla.hardware import create_cameras, create_robot
from rlinf.projects.fibocom_vla.hardware.base import JointRobotBase
from rlinf.projects.fibocom_vla.hardware.cameras import SynchronizedCameraRig
from rlinf.projects.fibocom_vla.hardware.dobot_nova import (
    DobotNovaRobot,
    _dobot_status_fault,
    _response_scalar,
)
from rlinf.projects.fibocom_vla.hardware.so101 import SO101Robot
from rlinf.projects.fibocom_vla.inference.rtc import PlannedChunk
from rlinf.projects.fibocom_vla.policy import MockChunkPolicy
from rlinf.projects.fibocom_vla.runtime import RealtimeController


class _PartialRobot(JointRobotBase):
    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self.resource_open = False
        self.disconnect_calls = 0

    def _connect_impl(self) -> None:
        self.resource_open = True
        raise RuntimeError("partial robot connect")

    def _read_state_impl(self) -> RobotState:
        raise AssertionError("state must not be read")

    def _send_joint_target_impl(self, target, period_s):
        raise AssertionError("target must not be sent")

    def _stop_impl(self) -> None:
        return None

    def _disconnect_impl(self) -> None:
        self.resource_open = False
        self.disconnect_calls += 1


def test_robot_connect_rolls_back_partially_open_backend() -> None:
    robot = _PartialRobot(
        RobotConfig(
            action_dim=1,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(0.1,),
        )
    )

    with pytest.raises(RuntimeError, match="partial robot connect"):
        robot.connect()

    assert not robot.resource_open
    assert robot.disconnect_calls == 1
    assert not robot.is_connected


class _PartialCamera:
    def __init__(self) -> None:
        self.connected = False
        self.disconnect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.connected = True
        raise RuntimeError("partial camera connect")

    def read(self, timeout_s: float = 1.0):
        raise AssertionError("frame must not be read")

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_calls += 1


def test_camera_rig_rolls_back_the_backend_that_raised() -> None:
    camera = _PartialCamera()
    rig = SynchronizedCameraRig({"partial": camera})

    with pytest.raises(RuntimeError, match="partial camera connect"):
        rig.connect()

    assert not camera.connected
    assert camera.disconnect_calls == 1


class _CloseFailPlanner:
    def __init__(self, policy, config, *, metrics) -> None:
        del config, metrics
        self.policy = policy
        self.observation = None

    def submit(self, request_id, observation, previous=None, executed_steps=0) -> None:
        del request_id, previous, executed_steps
        self.observation = observation

    def result(self) -> PlannedChunk:
        return PlannedChunk(0, self.policy.predict(self.observation), 0.0)

    def close(self) -> None:
        raise RuntimeError("planner close failed")


def test_planner_close_failure_cannot_skip_hardware_teardown(monkeypatch) -> None:
    from rlinf.projects.fibocom_vla.runtime import controller

    config = StackConfig()
    robot = create_robot(config.robot)
    cameras = SynchronizedCameraRig(create_cameras(config.cameras))
    policy = MockChunkPolicy(
        horizon=config.residual_rl.action_horizon,
        action_dim=config.robot.action_dim,
        period_s=1 / config.robot.control_hz,
    )
    monkeypatch.setattr(controller, "AsynchronousChunkPlanner", _CloseFailPlanner)

    with pytest.raises(RuntimeError, match="teardown failed"):
        RealtimeController(
            robot,
            cameras,
            policy,
            config.rtc,
            instruction="stack",
        ).run(maximum_control_steps=1)

    assert robot.safety.estopped
    assert not robot.is_connected
    assert all(not camera.is_connected for camera in cameras.cameras.values())


def test_cli_blocks_unvalidated_joint_limits_before_policy_loading(tmp_path) -> None:
    config = StackConfig(
        robot=RobotConfig(
            dry_run=False,
            action_adapter=ActionAdapterConfig(
                policy_dim=7,
                policy_names=(
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                    "aux",
                ),
                policy_units=("degree",) * 7,
                policy_state_dim=7,
                policy_state_names=(
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                    "aux",
                ),
                robot_units=("degree",) * 7,
                robot_from_policy=tuple(range(7)),
                scale=(1.0,) * 7,
                offset=(0.0,) * 7,
                validated=True,
                calibration_id="unit-test-identity-7d",
            ),
            options={"limits_require_hardware_validation": True},
        )
    )
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    args = argparse.Namespace(
        config=path,
        allow_motion=True,
        policy_factory="missing:factory",
        maximum_camera_skew_ms=35.0,
        instruction="stack",
        steps=1,
    )

    with pytest.raises(PermissionError, match="hardware-validated"):
        _run(args)


class _ConcurrentRobot(JointRobotBase):
    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self.send_entered = threading.Event()
        self.release_send = threading.Event()
        self.wire_events: list[str] = []

    def _connect_impl(self) -> None:
        return None

    def _read_state_impl(self) -> RobotState:
        return RobotState(np.zeros(1), ("joint",))

    def _send_joint_target_impl(self, target, period_s):
        del period_s
        self.send_entered.set()
        self.release_send.wait(timeout=2.0)
        self.wire_events.append("send")
        return target

    def _stop_impl(self) -> None:
        self.wire_events.append("stop")

    def _disconnect_impl(self) -> None:
        self.wire_events.append("disconnect")


def test_concurrent_stop_is_the_last_wire_command_after_inflight_send() -> None:
    robot = _ConcurrentRobot(
        RobotConfig(
            action_dim=1,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(1.0,),
            dry_run=False,
            action_adapter=ActionAdapterConfig(
                policy_dim=1,
                policy_names=("joint",),
                policy_units=("radian",),
                policy_state_dim=1,
                policy_state_names=("joint",),
                robot_units=("radian",),
                robot_from_policy=(0,),
                scale=(1.0,),
                offset=(0.0,),
                validated=True,
                calibration_id="unit-test-identity-1d",
            ),
        )
    )
    robot.connect()
    sender = threading.Thread(
        target=robot.send_joint_target,
        args=(np.asarray([0.5], dtype=np.float32), 0.05),
    )
    sender.start()
    assert robot.send_entered.wait(timeout=1.0)
    stopper = threading.Thread(target=robot.stop)
    stopper.start()
    robot.release_send.set()
    sender.join(timeout=2.0)
    stopper.join(timeout=2.0)

    assert robot.wire_events == ["send", "stop"]


class _DryRunWireProbe(JointRobotBase):
    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self.wire_events: list[str] = []

    def _connect_impl(self) -> None:
        self.wire_events.append("connect")

    def _read_state_impl(self) -> RobotState:
        return RobotState(np.zeros(1), ("joint",))

    def _send_joint_target_impl(self, target, period_s):
        del target, period_s
        self.wire_events.append("send")
        return np.zeros(1, dtype=np.float32)

    def _stop_impl(self) -> None:
        self.wire_events.append("hardware_stop")

    def _emergency_stop_impl(self) -> None:
        self.wire_events.append("hardware_emergency_stop")

    def _disconnect_impl(self) -> None:
        self.wire_events.append("disconnect")


def test_dry_run_cleanup_is_read_only_at_hardware_boundary() -> None:
    robot = _DryRunWireProbe(
        RobotConfig(
            action_dim=1,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(1.0,),
            dry_run=True,
        )
    )

    robot.connect()
    robot.stop()
    robot.emergency_stop()
    robot.disconnect()

    assert robot.wire_events == ["connect", "disconnect"]


def test_first_send_exception_uses_emergency_stop() -> None:
    class _SendRaisesAfterWire(_DryRunWireProbe):
        def _send_joint_target_impl(self, target, period_s):
            del target, period_s
            self.wire_events.append("write_started")
            raise RuntimeError("command acknowledgement parse failed")

    robot_config = RobotConfig(
        action_dim=1,
        control_hz=1000.0,
        joint_names=("joint",),
        joint_lower=(-1.0,),
        joint_upper=(1.0,),
        max_step=(1.0,),
        dry_run=False,
        action_adapter=ActionAdapterConfig(
            policy_dim=1,
            policy_names=("joint",),
            policy_units=("radian",),
            policy_state_dim=1,
            policy_state_names=("joint",),
            robot_units=("radian",),
            robot_from_policy=(0,),
            scale=(1.0,),
            offset=(0.0,),
            validated=True,
            calibration_id="unit-test-identity-1d",
        ),
    )
    stack_config = StackConfig(
        residual_rl=ResidualRLConfig(
            feature_dim=4,
            action_horizon=1,
            action_dim=1,
            actor_hidden_dim=4,
            actor_bottleneck_dim=2,
            target_actor_parameters=1,
            parameter_tolerance=10_000,
        ),
        rtc=RTCConfig(enabled=False, execution_horizon=1, overlap_horizon=1),
        robot=robot_config,
        cameras=(CameraConfig(width=4, height=4),),
    )
    robot = _SendRaisesAfterWire(robot_config)
    cameras = SynchronizedCameraRig(create_cameras(stack_config.cameras))
    policy = MockChunkPolicy(horizon=1, action_dim=1, period_s=0.001)

    with pytest.raises(RuntimeError, match="acknowledgement parse failed"):
        RealtimeController(
            robot,
            cameras,
            policy,
            stack_config.rtc,
            instruction="move",
        ).run(maximum_control_steps=1)

    assert robot.wire_events == [
        "connect",
        "write_started",
        "hardware_emergency_stop",
        "disconnect",
    ]


def test_robot_context_always_stops_before_disconnect() -> None:
    config = RobotConfig(
        action_dim=1,
        joint_names=("joint",),
        joint_lower=(-1.0,),
        joint_upper=(1.0,),
        max_step=(1.0,),
        dry_run=False,
        action_adapter=ActionAdapterConfig(
            policy_dim=1,
            policy_names=("joint",),
            policy_units=("radian",),
            policy_state_dim=1,
            policy_state_names=("joint",),
            robot_units=("radian",),
            robot_from_policy=(0,),
            scale=(1.0,),
            offset=(0.0,),
            validated=True,
            calibration_id="unit-test-identity-1d",
        ),
    )
    robot = _DryRunWireProbe(config)

    with robot:
        pass

    assert robot.wire_events == ["connect", "hardware_stop", "disconnect"]


def test_robot_context_reports_teardown_failure_on_normal_exit() -> None:
    class _StopFails(_DryRunWireProbe):
        def _stop_impl(self) -> None:
            raise RuntimeError("physical stop rejected")

    config = replace(
        RobotConfig(
            action_dim=1,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(1.0,),
        ),
        dry_run=False,
        action_adapter=ActionAdapterConfig(
            policy_dim=1,
            policy_names=("joint",),
            policy_units=("radian",),
            policy_state_dim=1,
            policy_state_names=("joint",),
            robot_units=("radian",),
            robot_from_policy=(0,),
            scale=(1.0,),
            offset=(0.0,),
            validated=True,
            calibration_id="unit-test-stop-failure",
        ),
    )

    with pytest.raises(RuntimeError, match="teardown failed"):
        with _StopFails(config):
            pass


def test_dobot_seventh_coordinate_requires_gripper_execution_backend() -> None:
    with pytest.raises(ConfigurationError, match="digital_output"):
        DobotNovaRobot(
            RobotConfig(
                backend="dobot_nova",
                action_dim=7,
                joint_names=(
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "gripper",
                ),
                joint_lower=(-180.0,) * 6 + (0.0,),
                joint_upper=(180.0,) * 6 + (1.0,),
                max_step=(5.0,) * 6 + (0.2,),
                options={"gripper_backend": "none"},
            )
        )


def test_dobot_gripper_configuration_fails_before_any_sdk_connection() -> None:
    joint_names = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper",
    )
    base = {
        "backend": "dobot_nova",
        "action_dim": 7,
        "joint_names": joint_names,
        "joint_lower": (-180.0,) * 6 + (0.0,),
        "joint_upper": (180.0,) * 6 + (1.0,),
        "max_step": (5.0,) * 6 + (0.2,),
    }
    with pytest.raises(ConfigurationError, match="digital_output"):
        DobotNovaRobot(RobotConfig(**base, options={"gripper_backend": "typo"}))
    with pytest.raises(ConfigurationError, match="gripper_do_index"):
        DobotNovaRobot(
            RobotConfig(
                **base,
                options={
                    "gripper_backend": "digital_output",
                    "gripper_do_index": 0,
                },
            )
        )
    with pytest.raises(ConfigurationError, match="max_feedback_age_s"):
        DobotNovaRobot(
            RobotConfig(
                **base,
                options={
                    "gripper_backend": "digital_output",
                    "max_feedback_age_s": float("nan"),
                },
            )
        )


def test_backend_units_are_bound_to_sdk_command_semantics() -> None:
    with pytest.raises(ConfigurationError, match="ServoJ/DO units"):
        DobotNovaRobot(
            RobotConfig(
                backend="dobot_nova",
                action_dim=7,
                joint_names=(
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "gripper",
                ),
                joint_lower=(-180.0,) * 6 + (0.0,),
                joint_upper=(180.0,) * 6 + (1.0,),
                max_step=(5.0,) * 6 + (1.0,),
                action_adapter=ActionAdapterConfig(
                    robot_units=("degree",) * 6 + ("normalized",)
                ),
                options={"gripper_backend": "digital_output"},
            )
        )
    with pytest.raises(ConfigurationError, match="LeRobot"):
        SO101Robot(
            RobotConfig(
                backend="so101",
                action_dim=6,
                joint_names=(
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ),
                joint_lower=(-100.0,) * 5 + (0.0,),
                joint_upper=(100.0,) * 6,
                max_step=(5.0,) * 6,
                action_adapter=ActionAdapterConfig(
                    robot_units=("radian",) * 5 + ("percent",)
                ),
                options={"use_degrees": True},
            )
        )


def test_dobot_status_checks_active_low_safety_and_approach_pause() -> None:
    safe = SimpleNamespace(
        collision_state=0,
        arm_approach_state=0,
        j4_approach_state=0,
        j5_approach_state=0,
        j6_approach_state=0,
        safety_state=0x63,
        has_error=lambda: False,
    )
    assert _dobot_status_fault(safe) is None
    safe.safety_state = 0x62
    assert "active-low" in _dobot_status_fault(safe)
    safe.safety_state = 0x63
    safe.j5_approach_state = 1
    assert "approach pause" in _dobot_status_fault(safe)


def test_dobot_scalar_response_parser_checks_error_and_value() -> None:
    assert _response_scalar("0,{1},DI(1);", "DI") == 1.0
    with pytest.raises(HardwareNotReadyError, match="failed with error"):
        _response_scalar("-2,{0},DI(1);", "DI")
