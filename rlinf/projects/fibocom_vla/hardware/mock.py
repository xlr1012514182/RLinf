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

"""Deterministic robot and camera backends for CPU integration tests."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..config import CameraConfig, RobotConfig
from ..contracts import RobotState
from ..errors import HardwareNotReadyError
from .base import JointRobotBase


class MockJointRobot(JointRobotBase):
    """In-memory joint robot with first-order target tracking."""

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self._position = (
            np.asarray(config.joint_lower, dtype=np.float32)
            + np.asarray(config.joint_upper, dtype=np.float32)
        ) * 0.5
        self._velocity = np.zeros(config.action_dim, dtype=np.float32)
        self.command_count = 0
        self.command_history: list[NDArray[np.float32]] = []

    def _connect_impl(self) -> None:
        return None

    def _read_state_impl(self) -> RobotState:
        return RobotState(
            joint_positions=self._position.copy(),
            joint_velocities=self._velocity.copy(),
            joint_names=tuple(self.config.joint_names),
        )

    def _send_joint_target_impl(
        self, target: NDArray[np.float32], period_s: float
    ) -> NDArray[np.float32]:
        previous = self._position.copy()
        self._position = target.copy()
        self._velocity = (self._position - previous) / period_s
        self.command_count += 1
        self.command_history.append(self._position.copy())
        return self._position.copy()

    def send_joint_target(
        self, target: NDArray[np.floating], period_s: float
    ) -> NDArray[np.float32]:
        """Execute commands even when the global config retains dry-run default."""

        if not self.is_connected:
            raise HardwareNotReadyError("mock robot is not connected")
        current = self.read_state().joint_positions
        safe = self.safety.filter(target, current)
        return self._send_joint_target_impl(safe, period_s)

    def _stop_impl(self) -> None:
        self._velocity.fill(0.0)

    def _disconnect_impl(self) -> None:
        return None


class MockCamera:
    """Synthetic RGB camera with frame index encoded in the first pixel."""

    def __init__(self, config: CameraConfig) -> None:
        config.validate()
        self.config = config
        self._connected = False
        self._frame_id = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def read(
        self, timeout_s: float = 1.0
    ) -> tuple[NDArray[np.uint8], int, Mapping[str, Any]]:
        if not self._connected:
            raise HardwareNotReadyError("mock camera is not connected")
        image = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        image[0, 0] = self._frame_id % 256
        timestamp_ns = time.monotonic_ns()
        metadata = {"frame_id": self._frame_id, "backend": "mock"}
        self._frame_id += 1
        return image, timestamp_ns, metadata

    def disconnect(self) -> None:
        self._connected = False
