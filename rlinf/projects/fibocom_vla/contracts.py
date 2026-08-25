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

"""Typed contracts shared by training, inference, and hardware backends."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .errors import ShapeMismatchError

FloatArray = NDArray[np.floating]


def _as_f32(value: Any, *, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ShapeMismatchError(f"{name} contains NaN or infinite values")
    return array


@dataclass(frozen=True)
class RobotState:
    """Timestamped proprioceptive state in a declared joint order."""

    joint_positions: NDArray[np.float32]
    joint_names: tuple[str, ...]
    timestamp_ns: int = field(default_factory=time.monotonic_ns)
    joint_velocities: NDArray[np.float32] | None = None
    gripper_position: float | None = None
    fault: str | None = None

    def __post_init__(self) -> None:
        positions = _as_f32(self.joint_positions, name="joint_positions")
        if positions.ndim != 1 or positions.shape[0] != len(self.joint_names):
            raise ShapeMismatchError(
                "joint_positions must be rank-1 and match joint_names"
            )
        object.__setattr__(self, "joint_positions", positions)
        if self.joint_velocities is not None:
            velocities = _as_f32(
                self.joint_velocities, name="joint_velocities"
            )
            if velocities.shape != positions.shape:
                raise ShapeMismatchError(
                    "joint_velocities must match joint_positions"
                )
            object.__setattr__(self, "joint_velocities", velocities)


@dataclass(frozen=True)
class Observation:
    """One synchronized policy observation.

    Images use HWC layout. ``timestamp_ns`` is monotonic host time; device
    timestamps remain in ``metadata`` so clock-domain conversion is explicit.
    """

    state: RobotState
    images: Mapping[str, NDArray[Any]]
    instruction: str
    timestamp_ns: int = field(default_factory=time.monotonic_ns)
    frame_id: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ShapeMismatchError("instruction must not be empty")
        normalized: dict[str, NDArray[Any]] = {}
        for name, image in self.images.items():
            array = np.asarray(image)
            if array.ndim not in (2, 3):
                raise ShapeMismatchError(
                    f"image {name!r} must be HW or HWC, got {array.shape}"
                )
            normalized[name] = array
        object.__setattr__(self, "images", normalized)


@dataclass(frozen=True)
class ActionChunk:
    """A time-indexed sequence of robot actions in ``[horizon, action_dim]``."""

    values: NDArray[np.float32]
    period_s: float
    source_observation_ns: int
    generated_ns: int = field(default_factory=time.monotonic_ns)
    committed_prefix: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _as_f32(self.values, name="action chunk")
        if values.ndim != 2 or min(values.shape) <= 0:
            raise ShapeMismatchError(
                f"action chunk must be non-empty [T, A], got {values.shape}"
            )
        if self.period_s <= 0:
            raise ShapeMismatchError("period_s must be positive")
        if not 0 <= self.committed_prefix <= values.shape[0]:
            raise ShapeMismatchError("committed_prefix is outside the action chunk")
        object.__setattr__(self, "values", values)

    @property
    def horizon(self) -> int:
        """Number of action steps in the chunk."""

        return int(self.values.shape[0])

    @property
    def action_dim(self) -> int:
        """Number of action coordinates per step."""

        return int(self.values.shape[1])

    def suffix(self, start: int) -> "ActionChunk":
        """Return the unexecuted suffix while retaining provenance."""

        if not 0 <= start < self.horizon:
            raise IndexError(f"suffix start {start} is outside horizon {self.horizon}")
        return ActionChunk(
            values=self.values[start:].copy(),
            period_s=self.period_s,
            source_observation_ns=self.source_observation_ns,
            committed_prefix=max(0, self.committed_prefix - start),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class PolicyOutput:
    """Action chunk plus inference-path evidence."""

    action: ActionChunk
    model_latency_ms: float
    path: str
    accepted_prefix: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_latency_ms < 0:
            raise ShapeMismatchError("model_latency_ms must be non-negative")


@runtime_checkable
class ChunkPolicy(Protocol):
    """Protocol implemented by base, draft, and accelerated policies."""

    def predict(self, observation: Observation) -> PolicyOutput:
        """Predict one action chunk."""


@runtime_checkable
class RobotBackend(Protocol):
    """Minimal robot backend used by the real-time controller."""

    @property
    def is_connected(self) -> bool:
        """Whether communication is active."""

    def connect(self) -> None:
        """Connect without sending an action."""

    def read_state(self) -> RobotState:
        """Read current proprioception and fault state."""

    def send_joint_target(self, target: FloatArray, period_s: float) -> FloatArray:
        """Send one bounded joint target and return what was actually sent."""

    def stop(self) -> None:
        """Stop motion using the safest backend-specific operation."""

    def disconnect(self) -> None:
        """Release the robot connection."""


@runtime_checkable
class CameraBackend(Protocol):
    """Timestamped camera backend contract."""

    @property
    def is_connected(self) -> bool:
        """Whether the camera stream is active."""

    def connect(self) -> None:
        """Open the camera stream."""

    def read(self, timeout_s: float = 1.0) -> tuple[NDArray[Any], int, Mapping[str, Any]]:
        """Return image, monotonic timestamp, and backend metadata."""

    def disconnect(self) -> None:
        """Close the stream."""
