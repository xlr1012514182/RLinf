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
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import (
    ActionAdapterConfig,
    CameraConfig,
    RobotConfig,
)
from rlinf.projects.fibocom_vla.errors import (
    ConfigurationError,
    HardwareNotReadyError,
)
from rlinf.projects.fibocom_vla.hardware.cameras import (
    RealSenseCamera,
    ROS2ImageCamera,
    SynchronizedCameraRig,
)
from rlinf.projects.fibocom_vla.hardware.ros2_joint import ROS2JointRobot


def test_ros_camera_rejects_stale_and_repeated_cached_frames() -> None:
    camera = ROS2ImageCamera(
        CameraConfig(
            backend="ros2_image",
            options={"max_frame_age_s": 0.01},
        )
    )
    camera._node = object()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    camera._latest = (
        image,
        time.monotonic_ns() - 1_000_000_000,
        {"backend": "test"},
    )
    with pytest.raises(HardwareNotReadyError, match="fresh"):
        camera.read(timeout_s=0.001)
    timestamp_ns = time.monotonic_ns()
    camera._latest = (image, timestamp_ns, {"backend": "test"})
    _, returned_ns, _ = camera.read(timeout_s=0.01)
    assert returned_ns == timestamp_ns
    with pytest.raises(HardwareNotReadyError, match="previously unseen"):
        camera.read(timeout_s=0.001)


def test_camera_rig_shares_one_timeout_budget_across_cameras() -> None:
    seen_timeouts: list[float] = []

    class _Camera:
        def __init__(self, delay_s: float) -> None:
            self.delay_s = delay_s

        def read(self, timeout_s: float):
            seen_timeouts.append(timeout_s)
            time.sleep(self.delay_s)
            return np.zeros((2, 2, 3)), time.monotonic_ns(), {}

    rig = SynchronizedCameraRig(
        {"first": _Camera(0.02), "second": _Camera(0.0)},
        maximum_skew_ms=100.0,
        maximum_attempts=1,
    )

    rig.read(timeout_s=0.05)

    assert len(seen_timeouts) == 2
    assert 0 < seen_timeouts[1] < seen_timeouts[0] - 0.01


def test_ros_joint_backend_rejects_stale_cached_state() -> None:
    robot = ROS2JointRobot(
        RobotConfig(
            backend="ros2_joint",
            action_dim=1,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(0.1,),
            options={"state_timeout_s": 0.001, "max_state_age_s": 0.01},
        )
    )
    robot._latest_state = (
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    robot._latest_state_ns = time.monotonic_ns() - 1_000_000_000

    with pytest.raises(HardwareNotReadyError, match="fresh"):
        robot._read_state_impl()


def test_ros_real_motion_requires_explicit_stop_endpoints() -> None:
    with pytest.raises(ConfigurationError, match="stop endpoints"):
        ROS2JointRobot(
            RobotConfig(
                backend="ros2_joint",
                action_dim=1,
                joint_names=("joint",),
                joint_lower=(-1.0,),
                joint_upper=(1.0,),
                max_step=(0.1,),
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


def test_ros_joint_units_must_match_declared_policy_robot_units() -> None:
    with pytest.raises(ConfigurationError, match="exactly match"):
        ROS2JointRobot(
            RobotConfig(
                backend="ros2_joint",
                action_dim=1,
                joint_names=("joint",),
                joint_lower=(-1.0,),
                joint_upper=(1.0,),
                max_step=(0.1,),
                action_adapter=ActionAdapterConfig(robot_units=("radian",)),
                options={"joint_units": ["meter"]},
            )
        )


def test_ros_stop_uses_acknowledged_service_not_cached_position() -> None:
    class _Trigger:
        class Request:
            pass

    class _Future:
        def done(self) -> bool:
            return True

        def result(self):
            return SimpleNamespace(success=True, message="stopped")

    class _Client:
        def call_async(self, request):
            assert isinstance(request, _Trigger.Request)
            return _Future()

    robot = ROS2JointRobot(
        RobotConfig(
            backend="ros2_joint",
            action_dim=1,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(0.1,),
            dry_run=True,
        )
    )
    robot._latest_state = (
        np.asarray([0.25], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    robot._latest_state_ns = time.monotonic_ns() - 1_000_000_000
    robot._trigger_type = _Trigger
    robot._stop_client = _Client()

    robot._stop_impl()


def test_ros_dobot_v4_emergency_service_sets_value_and_checks_res() -> None:
    class _EmergencyStop:
        class Request:
            def __init__(self) -> None:
                self.value = 0

    class _Future:
        def done(self) -> bool:
            return True

        def result(self):
            return SimpleNamespace(res=0)

    class _Client:
        def call_async(self, request):
            assert request.value == 1
            return _Future()

    robot = ROS2JointRobot(
        RobotConfig(
            backend="ros2_joint",
            action_dim=1,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(0.1,),
            action_adapter=ActionAdapterConfig(robot_units=("radian",)),
            options={
                "service_backend": "dobot_v4",
                "joint_units": ["radian"],
            },
        )
    )
    robot._emergency_stop_type = _EmergencyStop
    robot._emergency_stop_client = _Client()

    robot._emergency_stop_impl()


class _FakeColorFrame:
    def get_data(self):
        return np.zeros((2, 3, 3), dtype=np.uint8)

    def get_timestamp(self) -> float:
        return 12.5

    def get_frame_number(self) -> int:
        return 7


class _FakeFrames:
    def get_color_frame(self):
        return _FakeColorFrame()


class _FakePipeline:
    def __init__(self) -> None:
        self.started = False

    def start(self, configuration) -> None:
        del configuration
        self.started = True

    def wait_for_frames(self, timeout_ms: int):
        assert timeout_ms > 0
        return _FakeFrames()

    def stop(self) -> None:
        self.started = False


class _FakeRSConfig:
    def enable_device(self, serial: str) -> None:
        del serial

    def enable_stream(self, *args) -> None:
        del args


def test_realsense_rgb_only_path_never_constructs_or_processes_align(
    monkeypatch,
) -> None:
    fake_rs = SimpleNamespace(
        pipeline=_FakePipeline,
        config=_FakeRSConfig,
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(rgb8="rgb8", z16="z16"),
        align=lambda _stream: (_ for _ in ()).throw(
            AssertionError("RGB-only mode must not create rs.align")
        ),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    camera = RealSenseCamera(
        CameraConfig(
            backend="realsense",
            width=3,
            height=2,
            options={"enable_depth": False},
        )
    )

    camera.connect()
    image, _, metadata = camera.read()
    camera.disconnect()

    assert image.shape == (2, 3, 3)
    assert metadata["frame_number"] == 7
    assert camera._align is None
