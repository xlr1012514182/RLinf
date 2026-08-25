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

"""Generic ROS 2 joint-state and trajectory-command robot backend."""

from __future__ import annotations

import threading
import time

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import RobotState
from ..errors import HardwareNotReadyError, OptionalDependencyError
from .base import JointRobotBase


class ROS2JointRobot(JointRobotBase):
    """Use standard ``JointState`` and ``JointTrajectory`` messages.

    This generic backend is the fallback for robot vendors without a stable
    Python SDK. It intentionally avoids vendor services and can sit behind a
    ros2_control joint trajectory controller.
    """

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self._rclpy = None
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._publisher = None
        self._latest_state = None
        self._latest_state_ns = 0
        self._state_lock = threading.Lock()

    def _connect_impl(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory
        except ImportError as error:
            raise OptionalDependencyError(
                "ROS2 backend requires rclpy, sensor_msgs, and trajectory_msgs"
            ) from error
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        node_name = str(self.config.options.get("node_name", "fibocom_vla_robot"))
        self._node = rclpy.create_node(node_name)
        state_topic = str(self.config.options.get("joint_state_topic", "/joint_states"))
        command_topic = str(
            self.config.options.get(
                "command_topic", "/joint_trajectory_controller/joint_trajectory"
            )
        )
        self._node.create_subscription(JointState, state_topic, self._on_state, 10)
        self._publisher = self._node.create_publisher(
            JointTrajectory, command_topic, 10
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="ros2-joint-spin", daemon=True
        )
        self._spin_thread.start()

    def _on_state(self, message) -> None:
        mapping = {
            name: (position, velocity if index < len(message.velocity) else 0.0)
            for index, (name, position) in enumerate(zip(message.name, message.position))
            for velocity in [message.velocity[index] if index < len(message.velocity) else 0.0]
        }
        if not all(name in mapping for name in self.config.joint_names):
            return
        positions = np.asarray(
            [mapping[name][0] for name in self.config.joint_names], dtype=np.float32
        )
        velocities = np.asarray(
            [mapping[name][1] for name in self.config.joint_names], dtype=np.float32
        )
        with self._state_lock:
            self._latest_state = (positions, velocities)
            self._latest_state_ns = time.monotonic_ns()

    def _read_state_impl(self) -> RobotState:
        timeout_s = float(self.config.options.get("state_timeout_s", 1.0))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._state_lock:
                state = self._latest_state
                timestamp_ns = self._latest_state_ns
            if state is not None:
                return RobotState(
                    joint_positions=state[0].copy(),
                    joint_velocities=state[1].copy(),
                    joint_names=tuple(self.config.joint_names),
                    timestamp_ns=timestamp_ns,
                )
            time.sleep(0.005)
        raise HardwareNotReadyError("timed out waiting for ROS 2 JointState")

    def _send_joint_target_impl(
        self, target: NDArray[np.float32], period_s: float
    ) -> NDArray[np.float32]:
        if self._publisher is None:
            raise HardwareNotReadyError("ROS 2 trajectory publisher is unavailable")
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        message = JointTrajectory()
        message.joint_names = list(self.config.joint_names)
        point = JointTrajectoryPoint()
        point.positions = target.astype(float).tolist()
        total_ns = int(period_s * 1_000_000_000)
        point.time_from_start = Duration(
            sec=total_ns // 1_000_000_000,
            nanosec=total_ns % 1_000_000_000,
        )
        message.points = [point]
        self._publisher.publish(message)
        return target.copy()

    def _stop_impl(self) -> None:
        try:
            state = self._read_state_impl()
            self._send_joint_target_impl(state.joint_positions, 0.1)
        except Exception as error:
            raise HardwareNotReadyError("failed to publish ROS 2 hold command") from error

    def _disconnect_impl(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and bool(
            self.config.options.get("shutdown_rclpy", False)
        ):
            self._rclpy.shutdown()
        self._publisher = None
        self._node = None
        self._executor = None
        self._spin_thread = None
