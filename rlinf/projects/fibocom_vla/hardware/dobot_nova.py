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
_DOBOT_CRITICAL_ACTIVE_LOW_SAFETY_MASK = 0x63
_DOBOT_APPROACH_FIELDS = (
    "arm_approach_state",
    "j4_approach_state",
    "j5_approach_state",
    "j6_approach_state",
)


def _finite_option(
    options: dict,
    name: str,
    default: float,
    *,
    positive: bool = False,
) -> float:
    """Read and validate one finite numeric SDK option before any write."""

    try:
        value = float(options.get(name, default))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"robot.options.{name} must be numeric") from error
    if not np.isfinite(value) or (positive and value <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ConfigurationError(f"robot.options.{name} must be {qualifier}")
    return value


def _io_index(options: dict, name: str, default: int) -> int:
    """Validate a controller-local digital IO index."""

    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigurationError(f"robot.options.{name} must be an integer")
    value = int(value)
    if not 1 <= value <= 64:
        raise ConfigurationError(f"robot.options.{name} must be in [1, 64]")
    return value


def _dobot_status_fault(status: object) -> str | None:
    """Return a fail-closed fault string for one official V4 status packet.

    The official V4 feedback protocol documents the emergency, protective,
    system-emergency, and user-emergency bits as active-low.  The SDK's
    ``RobotStatus.is_ready`` does not inspect these bits or the four safety-skin
    approach-pause fields, so motion code must check them independently.
    """

    required = (
        "collision_state",
        "safety_state",
        *_DOBOT_APPROACH_FIELDS,
    )
    missing = [name for name in required if not hasattr(status, name)]
    if missing:
        return "Dobot status is missing safety fields: " + ", ".join(missing)
    has_error = getattr(status, "has_error", None)
    if not callable(has_error):
        return "Dobot status is missing has_error()"
    if bool(has_error()):
        return "Dobot controller reports an error"
    if bool(getattr(status, "collision_state")):
        return "Dobot collision state is active"
    paused = [name for name in _DOBOT_APPROACH_FIELDS if bool(getattr(status, name))]
    if paused:
        return "Dobot safety-skin approach pause is active: " + ", ".join(paused)
    safety_state = int(getattr(status, "safety_state")) & 0xFF
    inactive = (~safety_state) & _DOBOT_CRITICAL_ACTIVE_LOW_SAFETY_MASK
    if inactive:
        return (
            "Dobot active-low safety interlock is asserted "
            f"(SafetyState=0x{safety_state:02x}, unsafe_mask=0x{inactive:02x})"
        )
    return None


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


def _response_scalar(response: str, operation: str) -> float:
    """Parse one finite scalar from ``ErrorID,{value},Command();``."""

    _require_dobot_success(response, operation)
    match = re.search(r"\{\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\}", str(response))
    if match is None:
        raise HardwareNotReadyError(
            f"Dobot {operation} returned no scalar value: {response!r}"
        )
    value = float(match.group(1))
    if not np.isfinite(value):
        raise HardwareNotReadyError(f"Dobot {operation} returned a non-finite value")
    return value


class DobotNovaRobot(JointRobotBase):
    """Stream six Nova arm joints and optionally map a seventh value to DO."""

    def __init__(self, config: RobotConfig) -> None:
        if config.action_dim not in (6, 7):
            raise ConfigurationError("Dobot Nova action_dim must be 6 or 7")
        if tuple(config.joint_names[:6]) != NOVA_JOINTS:
            raise ConfigurationError(
                "Dobot Nova's first six names must be joint1 through joint6"
            )
        options = dict(config.options)
        maximum_age_s = _finite_option(
            options, "max_feedback_age_s", 0.25, positive=True
        )
        _finite_option(options, "initial_status_timeout_s", 3.0, positive=True)
        if config.action_dim == 7:
            gripper_backend = str(options.get("gripper_backend", "none"))
            feedback_backend = str(options.get("gripper_feedback_backend", "commanded"))
            if gripper_backend != "digital_output":
                raise ConfigurationError(
                    "a seven-dimensional Dobot config requires "
                    "gripper_backend='digital_output'"
                )
            if feedback_backend not in {"commanded", "digital_input"}:
                raise ConfigurationError(
                    "gripper_feedback_backend must be 'digital_input' or 'commanded'"
                )
            if not config.dry_run and feedback_backend != "digital_input":
                raise ConfigurationError(
                    "real seven-dimensional Dobot control requires independent "
                    "gripper_feedback_backend='digital_input'"
                )
            _io_index(options, "gripper_do_index", 1)
            if feedback_backend == "digital_input":
                _io_index(options, "gripper_di_index", 1)
            threshold = _finite_option(options, "gripper_threshold", 0.5)
            feedback_low = _finite_option(options, "gripper_feedback_low", 0.0)
            feedback_high = _finite_option(options, "gripper_feedback_high", 1.0)
            initial = _finite_option(options, "initial_gripper_state", 0.0)
            lower = float(config.joint_lower[6])
            upper = float(config.joint_upper[6])
            if not lower <= feedback_low < feedback_high <= upper:
                raise ConfigurationError(
                    "gripper feedback low/high must be ordered inside joint limits"
                )
            if feedback_low != 0.0 or feedback_high != 1.0:
                raise ConfigurationError(
                    "digital-output gripper feedback_low/high must be exactly 0/1"
                )
            if not lower < threshold < upper:
                raise ConfigurationError(
                    "gripper_threshold must lie strictly inside joint limits"
                )
            if not lower <= initial <= upper:
                raise ConfigurationError(
                    "initial_gripper_state must lie inside joint limits"
                )
            if initial not in {0.0, 1.0}:
                raise ConfigurationError(
                    "initial_gripper_state must be binary (0 or 1)"
                )
            if float(config.max_step[6]) < 1.0:
                raise ConfigurationError(
                    "binary Dobot gripper requires max_step[6] >= 1"
                )
        elif str(options.get("gripper_backend", "none")) != "none":
            raise ConfigurationError(
                "a six-dimensional Dobot config must use gripper_backend='none'"
            )
        expected_units = ("degree",) * 6
        if config.action_dim == 7:
            expected_units += ("binary",)
        declared_units = tuple(config.action_adapter.robot_units)
        if declared_units and declared_units != expected_units:
            raise ConfigurationError(
                "Dobot action_adapter.robot_units must match ServoJ/DO units: "
                f"{expected_units}"
            )
        if not config.dry_run and declared_units != expected_units:
            raise ConfigurationError(
                "real Dobot control requires explicit ServoJ/DO robot_units: "
                f"{expected_units}"
            )
        super().__init__(config)
        self._robot = None
        self._last_gripper = float(config.options.get("initial_gripper_state", 0.0))
        self._last_feedback_timestamp: int | None = None
        self._last_feedback_host_ns = 0
        self._maximum_feedback_age_s = maximum_age_s

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
        dry_run_writes = [
            name
            for name in (
                "request_control",
                "clear_error_on_connect",
                "enable_on_connect",
            )
            if bool(options.get(name, False))
        ]
        if self.config.dry_run and dry_run_writes:
            raise ConfigurationError(
                "Dobot dry-run is read-only; disable connection-time writes: "
                + ", ".join(dry_run_writes)
            )
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
        status_timestamp = int(status.timestamp)
        now_ns = time.monotonic_ns()
        if status_timestamp != self._last_feedback_timestamp:
            self._last_feedback_timestamp = status_timestamp
            self._last_feedback_host_ns = now_ns
        feedback_age_s = (now_ns - self._last_feedback_host_ns) / 1_000_000_000
        if feedback_age_s > self._maximum_feedback_age_s:
            raise HardwareNotReadyError(
                f"Dobot feedback is stale by {feedback_age_s:.3f}s"
            )
        positions = list(status.joint_state.q_actual)
        velocities = list(status.joint_state.qd_actual)
        if self.config.action_dim == 7:
            feedback_backend = str(
                self.config.options.get("gripper_feedback_backend", "commanded")
            )
            if feedback_backend == "digital_input":
                index = int(self.config.options.get("gripper_di_index", 1))
                raw_state = _response_scalar(self._robot.io.DI(index), "gripper DI")
                low = float(self.config.options.get("gripper_feedback_low", 0.0))
                high = float(self.config.options.get("gripper_feedback_high", 1.0))
                self._last_gripper = high if raw_state >= 0.5 else low
            elif feedback_backend != "commanded":
                raise ConfigurationError(
                    "gripper_feedback_backend must be 'digital_input' or 'commanded'"
                )
            positions.append(self._last_gripper)
            velocities.append(0.0)
        fault = _dobot_status_fault(status)
        return RobotState(
            joint_positions=np.asarray(positions, dtype=np.float32),
            joint_velocities=np.asarray(velocities, dtype=np.float32),
            joint_names=tuple(self.config.joint_names),
            timestamp_ns=now_ns,
            gripper_position=self._last_gripper
            if self.config.action_dim == 7
            else None,
            fault=fault,
        )

    def _send_joint_target_impl(
        self, target: NDArray[np.float32], period_s: float
    ) -> NDArray[np.float32]:
        if self._robot is None:
            raise HardwareNotReadyError("Dobot SDK object is unavailable")
        status = self._robot.GetStatus()
        if status is None:
            raise HardwareNotReadyError("Dobot produced no pre-command status")
        status_fault = _dobot_status_fault(status)
        if status_fault is not None:
            raise HardwareNotReadyError(status_fault)
        if not status.is_ready():
            raise HardwareNotReadyError("Dobot is not enabled and fault-free")
        if self.config.action_dim == 7 and float(target[6]) not in {0.0, 1.0}:
            raise HardwareNotReadyError(
                "Dobot digital-output gripper target must be exactly 0 or 1"
            )
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
        if backend != "digital_output":
            raise ConfigurationError(
                "gripper_backend must be 'digital_output' for a 7-D Dobot config"
            )
        index = int(self.config.options.get("gripper_do_index", 1))
        threshold = float(self.config.options.get("gripper_threshold", 0.5))
        response = self._robot.io.DO(index, int(value >= threshold))
        _require_dobot_success(response, "gripper DO")

    def _stop_impl(self) -> None:
        if self._robot is not None:
            response = self._robot.robot_control.Stop()
            _require_dobot_success(response, "Stop")

    def _emergency_stop_impl(self) -> None:
        if self._robot is not None:
            response = self._robot.robot_control.EmergencyStop(1)
            _require_dobot_success(response, "EmergencyStop")

    def _disconnect_impl(self) -> None:
        if self._robot is None:
            return
        try:
            if (
                bool(self.config.options.get("disable_on_disconnect", True))
                and not self.config.dry_run
            ):
                _require_dobot_success(
                    self._robot.robot_control.DisableRobot(), "DisableRobot"
                )
        finally:
            self._robot.Disconnect()
            self._robot = None
            self._last_feedback_timestamp = None
            self._last_feedback_host_ns = 0
