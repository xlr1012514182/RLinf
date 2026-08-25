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
        super().__init__(config)
        self._robot = None

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
        kwargs = {
            "port": str(port),
            "id": str(options.get("id", "fibocom-so101")),
            "disable_torque_on_disconnect": bool(
                options.get("disable_torque_on_disconnect", True)
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
        self._robot.connect(calibrate=bool(options.get("calibrate", False)))

    def _read_state_impl(self) -> RobotState:
        if self._robot is None:
            raise HardwareNotReadyError("SO101 SDK object is unavailable")
        observation = self._robot.get_observation()
        positions = np.asarray(
            [observation[f"{name}.pos"] for name in SO101_MOTORS],
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
            f"{name}.pos": float(observation[f"{name}.pos"])
            for name in SO101_MOTORS
        }
        self._robot.send_action(hold)
        if bool(self.config.options.get("stop_disables_torque", False)):
            self._robot.bus.disable_torque()

    def _disconnect_impl(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
