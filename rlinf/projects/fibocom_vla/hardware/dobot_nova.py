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

"""Dobot Nova/NovaLite adapter using the official TCP/IP Python V4 SDK."""

from __future__ import annotations

import re
import time

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import RobotState
from ..errors import ConfigurationError, HardwareNotReadyError, OptionalDependencyError
from .base import JointRobotBase

NOVA_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def _response_error_id(response: str) -> int:
    match = re.match(r"\s*(-?\d+)", str(response))
    if match is None:
        raise HardwareNotReadyError(f"unrecognized Dobot response: {response!r}")
    return int(match.group(1))


def _require_dobot_success(response: str, operation: str) -> None:
    error_id = _response_error_id(response)
    if error_id != 0:
        raise HardwareNotReadyError(
            f"Dobot {operation} failed with error {error_id}: {response}"
        )


class DobotNovaRobot(JointRobotBase):
    """Stream six Nova arm joints and optionally map a seventh value to DO."""

    def __init__(self, config: RobotConfig) -> None:
        if config.action_dim not in (6, 7):
            raise ConfigurationError("Dobot Nova action_dim must be 6 or 7")
        if tuple(config.joint_names[:6]) != NOVA_JOINTS:
            raise ConfigurationError(
                "Dobot Nova's first six names must be joint1 through joint6"
            )
        super().__init__(config)
        self._robot = None
        self._last_gripper = 0.0

    def _connect_impl(self) -> None:
        try:
            from dobot_sdk import DobotRobot
        except ImportError as error:
            raise OptionalDependencyError(
                "Dobot Nova requires the official Dobot-Arm/TCP-IP-Python-V4 SDK"
            ) from error
        options = dict(self.config.options)
        ip = options.get("ip")
        if not ip:
            raise ConfigurationError("robot.options.ip is required for Dobot Nova")
        self._robot = DobotRobot(
            str(ip),
            dashboard_port=int(options.get("dashboard_port", 29999)),
            feedback_port=int(options.get("feedback_port", 30004)),
            connect_timeout=float(options.get("connect_timeout", 5.0)),
            receive_timeout=float(options.get("receive_timeout", 10.0)),
        )
        self._robot.Connect()
        self._robot.StartFeedbackMonitor()
        if bool(options.get("request_control", False)):
            _require_dobot_success(
                self._robot.robot_control.RequestControl(), "RequestControl"
            )
        if bool(options.get("clear_error_on_connect", False)):
            _require_dobot_success(self._robot.robot_control.ClearError(), "ClearError")
        if bool(options.get("enable_on_connect", False)):
            if self.config.dry_run:
                raise ConfigurationError(
                    "enable_on_connect is forbidden while robot.dry_run=true"
                )
            _require_dobot_success(
                self._robot.robot_control.EnableRobot(
                    load=float(options.get("payload_kg", 0.0))
                ),
                "EnableRobot",
            )

    def _read_state_impl(self) -> RobotState:
        if self._robot is None:
            raise HardwareNotReadyError("Dobot SDK object is unavailable")
        timeout_s = float(self.config.options.get("initial_status_timeout_s", 3.0))
        deadline = time.monotonic() + timeout_s
        status = self._robot.GetStatus()
        while status is None and time.monotonic() < deadline:
            time.sleep(0.01)
            status = self._robot.GetStatus()
        if status is None:
            raise HardwareNotReadyError("Dobot feedback monitor produced no status")
        positions = list(status.joint_state.q_actual)
        velocities = list(status.joint_state.qd_actual)
        if self.config.action_dim == 7:
            positions.append(self._last_gripper)
            velocities.append(0.0)
        fault = None
        if status.has_error() or status.collision_state:
            fault = (
                f"Dobot fault: mode={status.robot_mode}, "
                f"collision={status.collision_state}, safety={status.safety_state}"
            )
        return RobotState(
            joint_positions=np.asarray(positions, dtype=np.float32),
            joint_velocities=np.asarray(velocities, dtype=np.float32),
            joint_names=tuple(self.config.joint_names),
            timestamp_ns=time.monotonic_ns(),
            gripper_position=self._last_gripper if self.config.action_dim == 7 else None,
            fault=fault,
        )

    def _send_joint_target_impl(
        self, target: NDArray[np.float32], period_s: float
    ) -> NDArray[np.float32]:
        if self._robot is None:
            raise HardwareNotReadyError("Dobot SDK object is unavailable")
        status = self._robot.GetStatus()
        if status is None or not status.is_ready():
            raise HardwareNotReadyError("Dobot is not enabled and fault-free")
        response = self._robot.motion.ServoJ(
            target[:6].tolist(),
            t=max(0.004, float(period_s)),
            aheadtime=float(self.config.options.get("aheadtime", 50.0)),
            gain=float(self.config.options.get("servo_gain", 500.0)),
        )
        _require_dobot_success(response, "ServoJ")
        if self.config.action_dim == 7:
            self._send_gripper(float(target[6]))
            self._last_gripper = float(target[6])
        return target.copy()

    def _send_gripper(self, value: float) -> None:
        backend = str(self.config.options.get("gripper_backend", "none"))
        if backend == "none":
            return
        if backend != "digital_output":
            raise ConfigurationError(
                "gripper_backend must be 'none' or 'digital_output'"
            )
        index = int(self.config.options.get("gripper_do_index", 1))
        threshold = float(self.config.options.get("gripper_threshold", 0.5))
        response = self._robot.io.DO(index, int(value >= threshold))
        _require_dobot_success(response, "gripper DO")

    def _stop_impl(self) -> None:
        if self._robot is not None:
            response = self._robot.robot_control.EmergencyStop(1)
            _require_dobot_success(response, "EmergencyStop")

    def _disconnect_impl(self) -> None:
        if self._robot is None:
            return
        try:
            if bool(self.config.options.get("disable_on_disconnect", True)) and not self.config.dry_run:
                _require_dobot_success(
                    self._robot.robot_control.DisableRobot(), "DisableRobot"
                )
        finally:
            self._robot.Disconnect()
            self._robot = None
