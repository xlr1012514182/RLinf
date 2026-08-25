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

"""SO-101 adapter backed by the official Hugging Face LeRobot SDK."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import RobotState
from ..errors import ConfigurationError, HardwareNotReadyError, OptionalDependencyError
from .base import JointRobotBase

SO101_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class SO101Robot(JointRobotBase):
    """Map the stack's ordered joint vector to LeRobot ``*.pos`` fields."""

    def __init__(self, config: RobotConfig) -> None:
        if config.action_dim != 6 or tuple(config.joint_names) != SO101_MOTORS:
            raise ConfigurationError(
                "SO101 requires six joints ordered as " + ", ".join(SO101_MOTORS)
            )
        use_degrees = bool(config.options.get("use_degrees", True))
        body_unit = "degree" if use_degrees else "normalized_-100_100"
        expected_units = (body_unit,) * 5 + ("percent",)
        declared_units = tuple(config.action_adapter.robot_units)
        if declared_units and declared_units != expected_units:
            raise ConfigurationError(
                "SO101 action_adapter.robot_units must match LeRobot's "
                f"use_degrees={use_degrees}: {expected_units}"
            )
        if not config.dry_run and declared_units != expected_units:
            raise ConfigurationError(
                "real SO101 control requires explicit LeRobot robot_units: "
                f"{expected_units}"
            )
        super().__init__(config)
        self._robot = None
        self._read_only_connection = False

    @staticmethod
    def _validate_calibration(calibration) -> None:
        if set(calibration) != set(SO101_MOTORS):
            raise ConfigurationError(
                "SO101 needs an existing six-motor LeRobot calibration; "
                "set robot.options.calibration_dir/id or run an explicitly "
                "authorized non-dry calibration first"
            )
        for expected_id, name in enumerate(SO101_MOTORS, start=1):
            value = calibration[name]
            if int(value.id) != expected_id:
                raise ConfigurationError(
                    f"SO101 calibration id for {name} is {value.id}, "
                    f"expected {expected_id}"
                )
            if int(value.range_min) >= int(value.range_max):
                raise ConfigurationError(
                    f"SO101 calibration range for {name} is invalid"
                )

    def _connect_impl(self) -> None:
        try:
            from lerobot.robots.so_follower import (
                SO101Follower,
                SO101FollowerConfig,
            )
        except ImportError as error:
            raise OptionalDependencyError(
                "SO101 requires LeRobot with the feetech extra: pip install 'lerobot[feetech]'"
            ) from error
        options = dict(self.config.options)
        port = options.get("port")
        if not port:
            raise ConfigurationError("robot.options.port is required for SO101")
        if self.config.dry_run and bool(options.get("calibrate", False)):
            raise ConfigurationError(
                "SO101 calibration writes motor registers and is forbidden in dry-run"
            )
        kwargs = {
            "port": str(port),
            "id": str(options.get("id", "fibocom-so101")),
            "disable_torque_on_disconnect": (
                False
                if self.config.dry_run
                else bool(options.get("disable_torque_on_disconnect", True))
            ),
            "max_relative_target": options.get(
                "max_relative_target", max(self.config.max_step)
            ),
            "use_degrees": bool(options.get("use_degrees", True)),
        }
        if options.get("calibration_dir"):
            kwargs["calibration_dir"] = Path(options["calibration_dir"])
        follower_config = SO101FollowerConfig(**kwargs)
        self._robot = SO101Follower(follower_config)
        calibrate = bool(options.get("calibrate", False))
        if not calibrate:
            self._validate_calibration(self._robot.calibration)
        if self.config.dry_run:
            # SOFollower.connect() always calls configure(), which disables
            # torque and writes operating/PID registers. A dry-run therefore
            # opens only the read bus and deliberately skips that high-level
            # lifecycle method.
            self._robot.bus.connect()
            self._read_only_connection = True
        else:
            self._robot.connect(calibrate=calibrate)
            self._validate_calibration(self._robot.calibration)
            self._read_only_connection = False

    def _read_state_impl(self) -> RobotState:
        if self._robot is None:
            raise HardwareNotReadyError("SO101 SDK object is unavailable")
        read_retries = int(self.config.options.get("read_retries", 3))
        if read_retries < 0:
            raise ConfigurationError("SO101 read_retries must be non-negative")
        raw_positions = self._robot.bus.sync_read(
            "Present_Position",
            num_retry=read_retries,
        )
        positions = np.asarray(
            [raw_positions[name] for name in SO101_MOTORS],
            dtype=np.float32,
        )
        return RobotState(
            joint_positions=positions,
            joint_names=SO101_MOTORS,
            timestamp_ns=time.monotonic_ns(),
            gripper_position=float(positions[-1]),
        )

    def _send_joint_target_impl(
        self, target: NDArray[np.float32], period_s: float
    ) -> NDArray[np.float32]:
        del period_s
        if self._robot is None:
            raise HardwareNotReadyError("SO101 SDK object is unavailable")
        sent = self._robot.send_action(
            {f"{name}.pos": float(value) for name, value in zip(SO101_MOTORS, target)}
        )
        return np.asarray(
            [sent[f"{name}.pos"] for name in SO101_MOTORS], dtype=np.float32
        )

    def _stop_impl(self) -> None:
        if self._robot is None:
            return
        observation = self._robot.get_observation()
        hold = {
            f"{name}.pos": float(observation[f"{name}.pos"]) for name in SO101_MOTORS
        }
        self._robot.send_action(hold)
        if bool(self.config.options.get("stop_disables_torque", False)):
            self._robot.bus.disable_torque()

    def _disconnect_impl(self) -> None:
        if self._robot is not None:
            try:
                if self._read_only_connection:
                    if self._robot.bus.is_connected:
                        self._robot.bus.disconnect(disable_torque=False)
                else:
                    self._robot.disconnect()
            finally:
                self._robot = None
                self._read_only_connection = False
