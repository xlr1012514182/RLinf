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

import time

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import (
    CameraConfig,
    ResidualRLConfig,
    RobotConfig,
    RTCConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.contracts import (
    ActionChunk,
    Observation,
    PolicyOutput,
)
from rlinf.projects.fibocom_vla.errors import SafetyViolationError
from rlinf.projects.fibocom_vla.hardware import create_cameras, create_robot
from rlinf.projects.fibocom_vla.hardware.cameras import SynchronizedCameraRig
from rlinf.projects.fibocom_vla.policy import MockChunkPolicy
from rlinf.projects.fibocom_vla.runtime import RealtimeController
from rlinf.projects.fibocom_vla.runtime.controller import ObservationSource


def test_mock_runtime_closes_finite_loop() -> None:
    config = StackConfig()
    robot = create_robot(config.robot)
    cameras = SynchronizedCameraRig(create_cameras(config.cameras))
    policy = MockChunkPolicy(
        horizon=config.residual_rl.action_horizon,
        action_dim=config.robot.action_dim,
        period_s=1 / config.robot.control_hz,
    )
    result = RealtimeController(
        robot,
        cameras,
        policy,
        config.rtc,
        instruction="stack the block",
    ).run(maximum_control_steps=5)
    assert result.control_steps == 5
    assert result.chunks == 1
    assert result.stop_reason == "maximum_control_steps"
    assert not robot.is_connected
    assert robot.safety.estopped


class _SequencePolicy:
    def __init__(self) -> None:
        self.call_index = 0

    def predict(self, observation: Observation) -> PolicyOutput:
        start = self.call_index * 10
        self.call_index += 1
        values = np.arange(start, start + 4, dtype=np.float32)[:, None]
        return PolicyOutput(
            action=ActionChunk(
                values=values,
                period_s=0.001,
                source_observation_ns=observation.timestamp_ns,
            ),
            model_latency_ms=0.0,
            path="sequence",
        )


def test_rtc_does_not_reexecute_prefix_completed_during_planning() -> None:
    config = StackConfig(
        residual_rl=ResidualRLConfig(
            feature_dim=4,
            action_horizon=4,
            action_dim=1,
            actor_hidden_dim=4,
            actor_bottleneck_dim=2,
            target_actor_parameters=51,
            parameter_tolerance=1,
        ),
        rtc=RTCConfig(
            execution_horizon=4,
            queue_threshold=2,
            overlap_horizon=4,
        ),
        robot=RobotConfig(
            action_dim=1,
            control_hz=1000.0,
            joint_names=("joint",),
            joint_lower=(-100.0,),
            joint_upper=(100.0,),
            max_step=(100.0,),
        ),
        cameras=(CameraConfig(width=4, height=4),),
    )
    robot = create_robot(config.robot)
    cameras = SynchronizedCameraRig(create_cameras(config.cameras))

    result = RealtimeController(
        robot,
        cameras,
        _SequencePolicy(),
        config.rtc,
        instruction="move",
    ).run(maximum_control_steps=6)

    assert result.control_steps == 6
    np.testing.assert_allclose(
        np.asarray(robot.command_history).squeeze(-1),
        [0.0, 1.0, 2.0, 3.0, 12.0, 13.0],
    )


def test_rtc_disabled_uses_synchronous_unconditioned_chunks() -> None:
    config = StackConfig(
        residual_rl=ResidualRLConfig(
            feature_dim=4,
            action_horizon=4,
            action_dim=1,
            actor_hidden_dim=4,
            actor_bottleneck_dim=2,
            target_actor_parameters=51,
            parameter_tolerance=1,
        ),
        rtc=RTCConfig(
            enabled=False,
            execution_horizon=4,
            queue_threshold=2,
            overlap_horizon=4,
        ),
        robot=RobotConfig(
            action_dim=1,
            control_hz=1000.0,
            joint_names=("joint",),
            joint_lower=(-100.0,),
            joint_upper=(100.0,),
            max_step=(100.0,),
        ),
        cameras=(CameraConfig(width=4, height=4),),
    )
    robot = create_robot(config.robot)
    cameras = SynchronizedCameraRig(create_cameras(config.cameras))
    policy = _SequencePolicy()

    result = RealtimeController(
        robot,
        cameras,
        policy,
        config.rtc,
        instruction="move",
    ).run(maximum_control_steps=6)

    assert result.control_steps == 6
    assert policy.call_index == 2
    np.testing.assert_allclose(
        np.asarray(robot.command_history).squeeze(-1),
        [0.0, 1.0, 2.0, 3.0, 10.0, 11.0],
    )


class _StalePolicy:
    def predict(self, observation: Observation) -> PolicyOutput:
        old_source_ns = time.monotonic_ns() - 2_000_000_000
        return PolicyOutput(
            action=ActionChunk(
                values=np.zeros((1, 1), dtype=np.float32),
                period_s=0.001,
                source_observation_ns=old_source_ns,
                generated_ns=old_source_ns + 1,
            ),
            model_latency_ms=0.0,
            path="stale",
        )


def test_controller_rejects_stale_action_before_hardware_send() -> None:
    config = StackConfig(
        residual_rl=ResidualRLConfig(
            feature_dim=4,
            action_horizon=1,
            action_dim=1,
            actor_hidden_dim=4,
            actor_bottleneck_dim=2,
            target_actor_parameters=1,
            parameter_tolerance=10_000,
        ),
        rtc=RTCConfig(
            enabled=False,
            execution_horizon=1,
            overlap_horizon=1,
        ),
        robot=RobotConfig(
            action_dim=1,
            control_hz=1000.0,
            maximum_source_age_s=0.01,
            maximum_generation_age_s=0.01,
            joint_names=("joint",),
            joint_lower=(-1.0,),
            joint_upper=(1.0,),
            max_step=(1.0,),
        ),
        cameras=(CameraConfig(width=4, height=4),),
    )
    robot = create_robot(config.robot)
    cameras = SynchronizedCameraRig(create_cameras(config.cameras))

    with pytest.raises(SafetyViolationError, match="stale"):
        RealtimeController(
            robot,
            cameras,
            _StalePolicy(),
            config.rtc,
            instruction="move",
        ).run(maximum_control_steps=1)

    assert robot.command_count == 0
    assert not robot.is_connected


def test_observation_source_rejects_state_camera_time_skew() -> None:
    config = RobotConfig(
        action_dim=1,
        joint_names=("joint",),
        joint_lower=(-1.0,),
        joint_upper=(1.0,),
        max_step=(1.0,),
        maximum_observation_age_s=1.0,
        maximum_state_camera_skew_ms=1.0,
    )

    class _Robot:
        def __init__(self) -> None:
            self.config = config

        def read_state(self):
            from rlinf.projects.fibocom_vla.contracts import RobotState

            return RobotState(
                np.zeros(1, dtype=np.float32),
                ("joint",),
                timestamp_ns=time.monotonic_ns() - 20_000_000,
            )

    class _Rig:
        def read(self, timeout_s: float):
            del timeout_s
            timestamp_ns = time.monotonic_ns()
            return (
                {"main": np.zeros((2, 2, 3), dtype=np.uint8)},
                timestamp_ns,
                {"camera_timestamps_ns": {"main": timestamp_ns}},
            )

    with pytest.raises(SafetyViolationError, match="state/camera skew"):
        ObservationSource(_Robot(), _Rig(), "move").capture()


class _FailingCamera:
    @property
    def is_connected(self) -> bool:
        return False

    def connect(self) -> None:
        raise RuntimeError("camera connect failed")

    def read(self, timeout_s: float = 1.0):
        del timeout_s
        raise AssertionError("read must not be called")

    def disconnect(self) -> None:
        return None


def test_camera_connect_failure_stops_and_disconnects_robot() -> None:
    config = StackConfig()
    robot = create_robot(config.robot)
    cameras = SynchronizedCameraRig({"broken": _FailingCamera()})
    policy = MockChunkPolicy(
        horizon=config.residual_rl.action_horizon,
        action_dim=config.robot.action_dim,
        period_s=1 / config.robot.control_hz,
    )

    with pytest.raises(RuntimeError, match="camera connect failed"):
        RealtimeController(
            robot,
            cameras,
            policy,
            config.rtc,
            instruction="stack",
        ).run(maximum_control_steps=1)

    assert not robot.is_connected
    assert robot.safety.estopped
