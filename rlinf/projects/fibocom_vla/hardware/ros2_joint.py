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

import math
import threading
import time

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import RobotState
from ..errors import ConfigurationError, HardwareNotReadyError, OptionalDependencyError
from .base import JointRobotBase


class ROS2JointRobot(JointRobotBase):
    """Use standard ``JointState`` and ``JointTrajectory`` messages.

    This generic backend is the fallback for robot vendors without a stable
    Python SDK. It intentionally avoids vendor services and can sit behind a
    ros2_control joint trajectory controller.
    """

    def __init__(self, config: RobotConfig) -> None:
        options = dict(config.options)
        service_backend = str(options.get("service_backend", "trigger"))
        if service_backend not in {"trigger", "dobot_v4"}:
            raise ConfigurationError(
                "ROS2 service_backend must be 'trigger' or 'dobot_v4'"
            )
        for name, default in (
            ("state_timeout_s", 1.0),
            ("max_state_age_s", 0.25),
            ("endpoint_preflight_timeout_s", 3.0),
            ("stop_timeout_s", 2.0),
        ):
            try:
                value = float(options.get(name, default))
            except (TypeError, ValueError) as error:
                raise ConfigurationError(
                    f"robot.options.{name} must be numeric"
                ) from error
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(
                    f"robot.options.{name} must be finite and positive"
                )
        joint_units = tuple(str(value) for value in options.get("joint_units", ()))
        if joint_units:
            if len(joint_units) != config.action_dim:
                raise ConfigurationError(
                    "robot.options.joint_units must match action_dim"
                )
            if any(unit not in {"radian", "meter"} for unit in joint_units):
                raise ConfigurationError(
                    "ROS2 JointTrajectory units must be 'radian' or 'meter'"
                )
            if joint_units != tuple(config.action_adapter.robot_units):
                raise ConfigurationError(
                    "ROS2 joint_units must exactly match action_adapter.robot_units"
                )
        if not config.dry_run:
            required_options = (
                "command_topic",
                "stop_service",
                "emergency_stop_service",
            )
            missing = [
                name for name in required_options if not config.options.get(name)
            ]
            if missing:
                raise ConfigurationError(
                    "real ROS2 joint control requires explicit command/stop endpoints: "
                    + ", ".join(missing)
                )
            placeholders = [
                name
                for name in required_options
                if "REPLACE" in str(config.options[name]).upper()
            ]
            if placeholders:
                raise ConfigurationError(
                    "real ROS2 endpoints still contain placeholders: "
                    + ", ".join(placeholders)
                )
            if not joint_units:
                raise ConfigurationError(
                    "real ROS2 joint control requires explicit joint_units"
                )
        super().__init__(config)
        self._service_backend = service_backend
        self._rclpy = None
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._publisher = None
        self._stop_client = None
        self._emergency_stop_client = None
        self._trigger_type = None
        self._stop_type = None
        self._emergency_stop_type = None
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
        with self._state_lock:
            self._latest_state = None
            self._latest_state_ns = 0
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
        stop_service = self.config.options.get("stop_service")
        emergency_service = self.config.options.get("emergency_stop_service")
        if self._service_backend == "trigger":
            try:
                from std_srvs.srv import Trigger
            except ImportError as error:
                raise OptionalDependencyError(
                    "ROS2 Trigger services require std_srvs"
                ) from error
            self._trigger_type = Trigger
            self._stop_type = Trigger
            self._emergency_stop_type = Trigger
        else:
            try:
                from dobot_msgs_v4.srv import EmergencyStop, Stop
            except ImportError as error:
                raise OptionalDependencyError(
                    "service_backend='dobot_v4' requires dobot_msgs_v4"
                ) from error
            self._stop_type = Stop
            self._emergency_stop_type = EmergencyStop
        if stop_service:
            self._stop_client = self._node.create_client(
                self._stop_type, str(stop_service)
            )
        if emergency_service:
            self._emergency_stop_client = self._node.create_client(
                self._emergency_stop_type, str(emergency_service)
            )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="ros2-joint-spin", daemon=True
        )
        self._spin_thread.start()
        if not self.config.dry_run:
            timeout_s = float(
                self.config.options.get("endpoint_preflight_timeout_s", 3.0)
            )
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                raise ConfigurationError(
                    "endpoint_preflight_timeout_s must be finite and positive"
                )
            deadline = time.monotonic() + timeout_s
            while (
                self._publisher.get_subscription_count() <= 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            if self._publisher.get_subscription_count() <= 0:
                raise HardwareNotReadyError(
                    f"no ROS2 trajectory subscriber on {command_topic}"
                )
            for name, client in (
                ("stop", self._stop_client),
                ("emergency stop", self._emergency_stop_client),
            ):
                if client is None or not client.wait_for_service(timeout_sec=timeout_s):
                    raise HardwareNotReadyError(
                        f"ROS2 {name} {self._service_backend} service is unavailable"
                    )

    def _on_state(self, message) -> None:
        mapping = {
            name: (position, velocity if index < len(message.velocity) else 0.0)
            for index, (name, position) in enumerate(
                zip(message.name, message.position)
            )
            for velocity in [
                message.velocity[index] if index < len(message.velocity) else 0.0
            ]
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
        maximum_age_s = float(self.config.options.get("max_state_age_s", 0.25))
        deadline = time.monotonic() + timeout_s
        last_age_s = float("inf")
        while time.monotonic() < deadline:
            with self._state_lock:
                state = self._latest_state
                timestamp_ns = self._latest_state_ns
            if state is not None:
                last_age_s = (time.monotonic_ns() - timestamp_ns) / 1_000_000_000
                if last_age_s <= maximum_age_s:
                    return RobotState(
                        joint_positions=state[0].copy(),
                        joint_velocities=state[1].copy(),
                        joint_names=tuple(self.config.joint_names),
                        timestamp_ns=timestamp_ns,
                    )
            time.sleep(0.005)
        raise HardwareNotReadyError(
            "timed out waiting for a fresh ROS 2 JointState; "
            f"last age={last_age_s:.3f}s"
        )

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

    def _call_service(
        self, client, service_type, operation: str, *, emergency: bool = False
    ) -> None:
        if client is None or service_type is None:
            raise HardwareNotReadyError(f"ROS2 {operation} service was not configured")
        timeout_s = float(self.config.options.get("stop_timeout_s", 2.0))
        request = service_type.Request()
        if self._service_backend == "dobot_v4" and emergency:
            request.value = 1
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not future.done():
            raise HardwareNotReadyError(f"ROS2 {operation} service timed out")
        response = future.result()
        if self._service_backend == "dobot_v4":
            accepted = response is not None and int(response.res) == 0
            message = "no response" if response is None else f"res={response.res}"
        else:
            accepted = response is not None and bool(response.success)
            message = "no response" if response is None else str(response.message)
        if not accepted:
            raise HardwareNotReadyError(
                f"ROS2 {operation} service rejected the request: {message}"
            )

    def _call_trigger(self, client, operation: str) -> None:
        """Compatibility helper for generic Trigger wrapper tests/users."""

        self._call_service(client, self._trigger_type, operation)

    def _stop_impl(self) -> None:
        # Never synthesize a hold from stale JointState: it could command a
        # reverse motion. Real mode is admitted only when this acknowledged
        # external stop primitive passed connection preflight.
        self._call_service(
            self._stop_client,
            self._stop_type or self._trigger_type,
            "stop",
        )

    def _emergency_stop_impl(self) -> None:
        self._call_service(
            self._emergency_stop_client,
            self._emergency_stop_type or self._trigger_type,
            "emergency stop",
            emergency=True,
        )

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
        self._stop_client = None
        self._emergency_stop_client = None
        self._trigger_type = None
        self._stop_type = None
        self._emergency_stop_type = None
        self._node = None
        self._executor = None
        self._spin_thread = None
        with self._state_lock:
            self._latest_state = None
            self._latest_state_ns = 0
