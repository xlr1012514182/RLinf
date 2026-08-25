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

        with self._command_lock:
            if self._connected:
                return
            try:
                self._connect_impl()
            except Exception:
                try:
                    self._disconnect_impl()
                except Exception:
                    logger.exception("robot connection rollback failed")
                self._connected = False
                raise
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

        if not np.isfinite(period_s) or period_s <= 0:
            raise ValueError("period_s must be finite and positive")
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
        """Latch software stop, then request a graceful backend halt.

        ``dry_run`` is a read-only hardware lifecycle: it may connect and read
        state, but it must not send even a hold/stop command to an actuator.
        """

        self.safety.engage_estop("stop requested")
        with self._command_lock:
            if self._connected and not self.config.dry_run:
                self._stop_impl()
            elif self._connected:
                logger.info("dry-run: suppressing hardware stop command")

    def emergency_stop(self) -> None:
        """Latch software stop and invoke the backend emergency path.

        No emergency command is sent in ``dry_run`` because that mode never
        authorizes an actuator write in the first place.
        """

        self.safety.engage_estop("emergency stop requested")
        with self._command_lock:
            if self._connected and not self.config.dry_run:
                self._emergency_stop_impl()
            elif self._connected:
                logger.info("dry-run: suppressing hardware emergency-stop command")

    def disconnect(self) -> None:
        """Release the connection idempotently."""

        with self._command_lock:
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
        cleanup_errors: list[Exception] = []
        try:
            if self._connected:
                if exc_value is None:
                    self.stop()
                else:
                    self.emergency_stop()
        except Exception as error:
            cleanup_errors.append(error)
            logger.exception("robot stop failed during context-manager teardown")
        try:
            self.disconnect()
        except Exception as error:
            cleanup_errors.append(error)
            logger.exception("robot disconnect failed during context-manager teardown")
        if cleanup_errors and exc_value is None:
            raise RuntimeError(
                "robot context-manager teardown failed"
            ) from cleanup_errors[0]

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

    def _emergency_stop_impl(self) -> None:
        """Default emergency path for backends without a stronger primitive."""

        self._stop_impl()

    @abstractmethod
    def _disconnect_impl(self) -> None: ...
