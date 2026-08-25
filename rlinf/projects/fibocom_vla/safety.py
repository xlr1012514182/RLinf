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

"""Backend-independent joint action safety gates."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import SafetyViolationError, ShapeMismatchError


@dataclass(frozen=True)
class JointSafetyEnvelope:
    """Position and per-command delta limits in backend units."""

    lower: NDArray[np.float32]
    upper: NDArray[np.float32]
    max_step: NDArray[np.float32]

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        max_step = np.asarray(self.max_step, dtype=np.float32)
        if lower.ndim != 1 or upper.shape != lower.shape or max_step.shape != lower.shape:
            raise ShapeMismatchError("safety vectors must be equal rank-1 arrays")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ShapeMismatchError("joint limits must be finite")
        if np.any(lower >= upper) or np.any(max_step <= 0):
            raise ShapeMismatchError("invalid joint range or max_step")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "max_step", max_step)


class ActionSafetyGate:
    """Validate and rate-limit joint targets with a software e-stop latch."""

    def __init__(self, envelope: JointSafetyEnvelope) -> None:
        self.envelope = envelope
        self._estop_reason: str | None = None
        self._lock = threading.Lock()

    @property
    def estopped(self) -> bool:
        """Whether the software stop latch is active."""

        with self._lock:
            return self._estop_reason is not None

    @property
    def estop_reason(self) -> str | None:
        """Return the latched reason, if any."""

        with self._lock:
            return self._estop_reason

    def engage_estop(self, reason: str) -> None:
        """Latch the gate closed until explicitly reset."""

        with self._lock:
            self._estop_reason = reason or "unspecified software e-stop"

    def reset_estop(self) -> None:
        """Reset only the software latch; hardware faults remain backend-owned."""

        with self._lock:
            self._estop_reason = None

    def filter(
        self,
        target: NDArray[np.floating],
        current: NDArray[np.floating],
        *,
        clip: bool = True,
    ) -> NDArray[np.float32]:
        """Validate a target, then clip it to position and delta limits.

        Args:
            target: Requested joint target.
            current: Most recently observed joint position.
            clip: If false, any limit violation raises instead of clipping.

        Returns:
            The command that is safe to send.
        """

        reason = self.estop_reason
        if reason is not None:
            raise SafetyViolationError(f"software e-stop is active: {reason}")
        target_array = np.asarray(target, dtype=np.float32)
        current_array = np.asarray(current, dtype=np.float32)
        if target_array.shape != self.envelope.lower.shape:
            raise ShapeMismatchError(
                f"target shape {target_array.shape} does not match safety envelope"
            )
        if current_array.shape != target_array.shape:
            raise ShapeMismatchError("current and target shapes differ")
        if not np.all(np.isfinite(target_array)) or not np.all(np.isfinite(current_array)):
            self.engage_estop("non-finite action or state")
            raise SafetyViolationError("non-finite action or state")
        bounded = np.clip(target_array, self.envelope.lower, self.envelope.upper)
        delta = np.clip(
            bounded - current_array,
            -self.envelope.max_step,
            self.envelope.max_step,
        )
        safe = np.clip(
            current_array + delta,
            self.envelope.lower,
            self.envelope.upper,
        )
        if not clip and not np.array_equal(safe, target_array):
            raise SafetyViolationError("target exceeds a configured safety limit")
        return safe.astype(np.float32, copy=False)
