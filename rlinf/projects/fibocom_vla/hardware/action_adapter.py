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

"""Explicit policy-space to robot-space action and state adaptation.

This module is deliberately independent of policy loading, controller logic,
and backend SDKs. It converts declared policy semantics into absolute robot
joint targets *before* joint/delta limit checks. It never assumes that a
checkpoint-native or end-effector action can be sent to a joint backend.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import ActionChunk, Observation, RobotState
from ..errors import ConfigurationError, SafetyViolationError, ShapeMismatchError


def _finite_f32(value: Any, *, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ShapeMismatchError(f"{name} contains NaN or infinite values")
    return array


class PolicyRobotActionAdapter:
    """Convert a declared joint-policy contract into robot joint targets.

    The calibrated relation is ``q_robot[r] = scale[r] * q_policy[p] +
    offset[r]``, where ``p = robot_from_policy[r]``. Negative scale is allowed
    for reversed joint direction. The inverse relation is used for policy
    observations and velocities.

    Supported policy action modes:

    - ``absolute``: each row is an absolute policy-space target.
    - ``delta_from_observation``: every row is relative to the source
      observation, not the preceding row.
    - ``integrated_delta``: rows are accumulated from the source observation.
    - ``velocity``: rows are integrated with the declared control period.

    Only affine ``joint`` action/state frames are implemented. Cartesian and
    checkpoint-native contracts require a reviewed IK/FK adapter and fail
    closed here.
    """

    _PERIOD_REL_TOLERANCE = 1e-6
    _PERIOD_ABS_TOLERANCE = 1e-9
    _LIMIT_TOLERANCE = 1e-6

    def __init__(self, robot_config: RobotConfig) -> None:
        robot_config.validate()
        self.robot_config = robot_config
        self.config = robot_config.action_adapter
        if self.config.implementation != "generic_affine_joint":
            raise ConfigurationError(
                "PolicyRobotActionAdapter only implements generic_affine_joint; "
                "load the declared custom module:function adapter instead"
            )
        self.robot_dim = robot_config.action_dim
        self.policy_dim = self.config.resolved_policy_dim(self.robot_dim)
        self.policy_state_dim = self.config.resolved_policy_state_dim(self.robot_dim)
        if self.config.policy_frame != "joint":
            raise ConfigurationError(
                f"generic adapter cannot convert {self.config.policy_frame!r} "
                "actions to absolute robot joints; reviewed IK is required"
            )
        if self.config.policy_state_frame != "joint":
            raise ConfigurationError(
                f"generic adapter cannot derive {self.config.policy_state_frame!r} "
                "state from robot joints; reviewed FK/state mapping is required"
            )
        if not (self.policy_dim == self.policy_state_dim == self.robot_dim):
            raise ConfigurationError(
                "generic affine adapter requires equal policy/state/robot dimensions"
            )

        mapping = self.config.robot_from_policy or tuple(range(self.robot_dim))
        if sorted(mapping) != list(range(self.policy_dim)):
            raise ConfigurationError(
                "robot_from_policy must be a complete permutation for inversion"
            )
        self.robot_from_policy = np.asarray(mapping, dtype=np.int64)
        self.scale = _finite_f32(
            self.config.scale or (1.0,) * self.robot_dim,
            name="calibration scale",
        )
        self.offset = _finite_f32(
            self.config.offset or (0.0,) * self.robot_dim,
            name="calibration offset",
        )
        if np.any(self.scale == 0):
            raise ConfigurationError("calibration scale must be non-zero")
        self.lower = _finite_f32(robot_config.joint_lower, name="joint lower")
        self.upper = _finite_f32(robot_config.joint_upper, name="joint upper")
        self.max_step = _finite_f32(robot_config.max_step, name="joint max_step")
        self.robot_joint_names = tuple(robot_config.joint_names)
        self.policy_names = self.config.policy_names or self._derived_policy_names()
        self.policy_state_names = self.config.policy_state_names or self.policy_names
        if tuple(self.policy_state_names) != tuple(self.policy_names):
            raise ConfigurationError(
                "joint policy_state_names must match policy_names for inversion"
            )
        self.expected_period_s = 1.0 / float(robot_config.control_hz)

    def _derived_policy_names(self) -> tuple[str, ...]:
        names = [""] * self.policy_dim
        for robot_index, policy_index in enumerate(self.robot_from_policy):
            names[int(policy_index)] = self.robot_joint_names[robot_index]
        return tuple(names)

    @staticmethod
    def _validate_timestamp(value: int, *, name: str) -> None:
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ShapeMismatchError(f"{name} must be a positive monotonic timestamp")

    def _validate_period(self, period_s: float) -> None:
        if not math.isfinite(period_s) or period_s <= 0:
            raise ShapeMismatchError("action period_s must be finite and positive")
        if not math.isclose(
            period_s,
            self.expected_period_s,
            rel_tol=self._PERIOD_REL_TOLERANCE,
            abs_tol=self._PERIOD_ABS_TOLERANCE,
        ):
            raise ShapeMismatchError(
                f"action period_s={period_s} does not match configured "
                f"control period {self.expected_period_s}"
            )

    def _validate_robot_state(self, state: RobotState) -> NDArray[np.float32]:
        if tuple(state.joint_names) != self.robot_joint_names:
            raise ShapeMismatchError(
                "robot state joint_names do not match the calibrated robot order"
            )
        self._validate_timestamp(state.timestamp_ns, name="robot state timestamp_ns")
        positions = _finite_f32(state.joint_positions, name="robot positions")
        if positions.shape != (self.robot_dim,):
            raise ShapeMismatchError(
                f"robot positions must have shape ({self.robot_dim},)"
            )
        if np.any(positions < self.lower) or np.any(positions > self.upper):
            raise SafetyViolationError(
                "observed robot state is outside calibrated joint limits"
            )
        return positions

    def _robot_to_policy(
        self, robot_values: NDArray[np.floating], *, apply_offset: bool
    ) -> NDArray[np.float32]:
        values = _finite_f32(robot_values, name="robot state")
        if values.shape != (self.robot_dim,):
            raise ShapeMismatchError(f"robot state must have shape ({self.robot_dim},)")
        calibrated = values - self.offset if apply_offset else values
        calibrated = calibrated / self.scale
        policy = np.empty(self.policy_dim, dtype=np.float32)
        policy[self.robot_from_policy] = calibrated
        return policy

    def robot_state_to_policy(self, state: RobotState) -> RobotState:
        """Invert mapping/calibration and preserve timestamp/fault provenance."""

        positions = self._validate_robot_state(state)
        policy_positions = self._robot_to_policy(positions, apply_offset=True)
        policy_velocities = None
        if state.joint_velocities is not None:
            policy_velocities = self._robot_to_policy(
                state.joint_velocities, apply_offset=False
            )
        policy_gripper = None
        gripper_indices = [
            index
            for index, name in enumerate(self.robot_joint_names)
            if "gripper" in name.lower()
        ]
        if len(gripper_indices) == 1:
            policy_index = int(self.robot_from_policy[gripper_indices[0]])
            policy_gripper = float(policy_positions[policy_index])
        return RobotState(
            joint_positions=policy_positions,
            joint_names=tuple(self.policy_state_names),
            timestamp_ns=int(state.timestamp_ns),
            joint_velocities=policy_velocities,
            gripper_position=policy_gripper,
            fault=state.fault,
        )

    def observation_to_policy_state(self, observation: Observation) -> RobotState:
        """Validate required cameras and return the reverse-adapted state."""

        missing = sorted(set(self.config.required_cameras) - set(observation.images))
        if missing:
            raise ShapeMismatchError(
                "observation is missing required policy cameras: " + ", ".join(missing)
            )
        self._validate_timestamp(
            observation.timestamp_ns, name="observation timestamp_ns"
        )
        return self.robot_state_to_policy(observation.state)

    def _policy_absolute_targets(
        self,
        values: NDArray[np.float32],
        observed_policy_positions: NDArray[np.float32],
        period_s: float,
    ) -> NDArray[np.float32]:
        coordinate_modes = (
            self.config.coordinate_modes or (self.config.action_mode,) * self.policy_dim
        )
        absolute = np.empty_like(values)
        for index, mode in enumerate(coordinate_modes):
            coordinate = values[:, index]
            observed = observed_policy_positions[index]
            if mode == "absolute":
                absolute[:, index] = coordinate
            elif mode == "delta_from_observation":
                absolute[:, index] = observed + coordinate
            elif mode == "integrated_delta":
                absolute[:, index] = observed + np.cumsum(coordinate)
            elif mode == "velocity":
                absolute[:, index] = observed + np.cumsum(
                    coordinate * np.float32(period_s)
                )
            else:
                raise AssertionError(f"unreachable action mode: {mode}")
        return absolute

    def _policy_to_robot_targets(
        self, policy_absolute: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        reordered = policy_absolute[:, self.robot_from_policy]
        return (reordered * self.scale[None, :] + self.offset[None, :]).astype(
            np.float32,
            copy=False,
        )

    def _validate_robot_targets(
        self,
        targets: NDArray[np.float32],
        observed_robot_positions: NDArray[np.float32],
    ) -> None:
        if np.any(targets < self.lower[None, :] - self._LIMIT_TOLERANCE) or np.any(
            targets > self.upper[None, :] + self._LIMIT_TOLERANCE
        ):
            raise SafetyViolationError(
                "semantically converted target exceeds calibrated joint limits"
            )
        previous = np.concatenate(
            (observed_robot_positions[None, :], targets[:-1]), axis=0
        )
        if np.any(
            np.abs(targets - previous) > self.max_step[None, :] + self._LIMIT_TOLERANCE
        ):
            raise SafetyViolationError(
                "semantically converted target exceeds per-command max_step"
            )

    def adapt_action_chunk(
        self, chunk: ActionChunk, observation: Observation
    ) -> ActionChunk:
        """Convert one policy chunk to checked absolute robot joint targets."""

        if not self.robot_config.dry_run and not self.config.validated:
            raise PermissionError("real motion requires action_adapter.validated=true")
        self._validate_period(chunk.period_s)
        self._validate_timestamp(
            chunk.source_observation_ns, name="action source_observation_ns"
        )
        self._validate_timestamp(chunk.generated_ns, name="action generated_ns")
        if chunk.source_observation_ns != observation.timestamp_ns:
            raise ShapeMismatchError(
                "action chunk is not anchored to the supplied observation"
            )
        if chunk.generated_ns < chunk.source_observation_ns:
            raise ShapeMismatchError(
                "action generated_ns predates its source observation"
            )
        if chunk.action_dim != self.policy_dim:
            raise ShapeMismatchError(
                f"policy action dim {chunk.action_dim} does not match "
                f"declared dimension {self.policy_dim}"
            )
        values = _finite_f32(chunk.values, name="policy action chunk")
        policy_state = self.observation_to_policy_state(observation)
        policy_absolute = self._policy_absolute_targets(
            values, policy_state.joint_positions, chunk.period_s
        )
        robot_targets = self._policy_to_robot_targets(policy_absolute)
        observed_robot_positions = self._validate_robot_state(observation.state)
        self._validate_robot_targets(robot_targets, observed_robot_positions)
        metadata = dict(chunk.metadata)
        metadata["policy_robot_adapter"] = {
            "action_mode": self.config.action_mode,
            "policy_frame": self.config.policy_frame,
            "calibration_validated": self.config.validated,
            "calibration_id": self.config.calibration_id,
            "output_semantics": "absolute_robot_joint_target",
        }
        return ActionChunk(
            values=robot_targets,
            period_s=chunk.period_s,
            source_observation_ns=chunk.source_observation_ns,
            generated_ns=chunk.generated_ns,
            committed_prefix=chunk.committed_prefix,
            metadata=metadata,
        )
