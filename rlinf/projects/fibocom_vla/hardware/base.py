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

"""Shared lifecycle and safety behavior for joint-space robot backends."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import RobotState
from ..errors import HardwareNotReadyError
from ..safety import ActionSafetyGate, JointSafetyEnvelope

logger = logging.getLogger(__name__)


class JointRobotBase(ABC):
    """Common lifecycle, dry-run, fault, and joint-limit enforcement."""

    def __init__(self, config: RobotConfig) -> None:
        config.validate()
        self.config = config
        self.safety = ActionSafetyGate(
            JointSafetyEnvelope(
                lower=np.asarray(config.joint_lower, dtype=np.float32),
                upper=np.asarray(config.joint_upper, dtype=np.float32),
                max_step=np.asarray(config.max_step, dtype=np.float32),
            )
        )
        self._connected = False
        self._command_lock = threading.Lock()
        self._last_state: RobotState | None = None

    @property
    def is_connected(self) -> bool:
        """Whether the backend connection is active."""

        return self._connected

    def connect(self) -> None:
        """Connect once without issuing a target command."""

        if self._connected:
            return
        self._connect_impl()
        self._connected = True

    def read_state(self) -> RobotState:
        """Read and validate current state, latching faults in the safety gate."""

        if not self._connected:
            raise HardwareNotReadyError("robot is not connected")
        state = self._read_state_impl()
        if state.fault:
            self.safety.engage_estop(state.fault)
        self._last_state = state
        return state

    def send_joint_target(
        self, target: NDArray[np.floating], period_s: float
    ) -> NDArray[np.float32]:
        """Filter one target and either log dry-run or send it atomically."""

        if period_s <= 0:
            raise ValueError("period_s must be positive")
        if not self._connected:
            raise HardwareNotReadyError("robot is not connected")
        with self._command_lock:
            current = self.read_state().joint_positions
            safe = self.safety.filter(target, current)
            if self.config.dry_run:
                logger.info("dry-run joint target: %s", safe.tolist())
                return safe
            actual = np.asarray(
                self._send_joint_target_impl(safe, period_s), dtype=np.float32
            )
            if actual.shape != safe.shape:
                self.safety.engage_estop("backend returned an invalid command shape")
                raise HardwareNotReadyError("backend returned an invalid command shape")
            return actual

    def stop(self) -> None:
        """Latch software stop, then invoke the backend's safest stop operation."""

        self.safety.engage_estop("stop requested")
        if self._connected:
            self._stop_impl()

    def disconnect(self) -> None:
        """Release the connection idempotently."""

        if not self._connected:
            return
        try:
            self._disconnect_impl()
        finally:
            self._connected = False

    def __enter__(self) -> "JointRobotBase":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is not None:
            try:
                self.stop()
            except Exception:
                logger.exception("robot stop failed while handling another exception")
        self.disconnect()

    @abstractmethod
    def _connect_impl(self) -> None: ...

    @abstractmethod
    def _read_state_impl(self) -> RobotState: ...

    @abstractmethod
    def _send_joint_target_impl(
        self, target: NDArray[np.float32], period_s: float
    ) -> NDArray[np.float32]: ...

    @abstractmethod
    def _stop_impl(self) -> None: ...

    @abstractmethod
    def _disconnect_impl(self) -> None: ...
