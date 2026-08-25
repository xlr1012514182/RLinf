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

"""Fail-closed API boundary for deployment-specific IK/FK implementations.

The repository cannot infer a safe Cartesian-to-joint mapping for an SO101,
Dobot Nova, or another arm from a VLA checkpoint contract alone.  This module
therefore loads a user-supplied kinematics engine and validates both sides of
that engine.  The engine consumes checkpoint-native observations/actions and
must return absolute robot-joint targets; no target is silently clipped.

Set ``robot.options.kinematics_factory`` to ``"module:function"``.  The
factory receives :class:`~rlinf.projects.fibocom_vla.config.RobotConfig` and
must return an object implementing both of these callables::

    encode_policy_state(observation) -> array | RobotState
    policy_chunk_to_robot_targets(
        observation, policy_values, period_s, action_adapter_config
    ) -> array[T, robot_action_dim]

Native RTC is deliberately disabled because a general IK mapping need not be
invertible or single-valued.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import ActionChunk, Observation, RobotState
from ..errors import ConfigurationError, SafetyViolationError, ShapeMismatchError

_PERIOD_REL_TOLERANCE = 1e-6
_PERIOD_ABS_TOLERANCE = 1e-9
_LIMIT_TOLERANCE = 1e-6
_PLACEHOLDER_MARKERS = ("REPLACE", "PLACEHOLDER", "TODO", "MODULE:FUNCTION")


def _finite_f32(value: Any, *, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ShapeMismatchError(f"{name} contains NaN or infinite values")
    return array


def _positive_timestamp(value: Any, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ShapeMismatchError(f"{name} must be a positive monotonic timestamp")
    return int(value)


def _load_engine(robot_config: RobotConfig) -> tuple[Any, str]:
    value = robot_config.options.get("kinematics_factory")
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            "robot.options.kinematics_factory must name a reviewed module:function"
        )
    qualified_name = value.strip()
    uppercase = qualified_name.upper()
    if any(marker in uppercase for marker in _PLACEHOLDER_MARKERS):
        raise ConfigurationError(
            "robot.options.kinematics_factory is still a placeholder"
        )
    module_name, separator, factory_name = qualified_name.partition(":")
    if not separator or not module_name.strip() or not factory_name.strip():
        raise ConfigurationError(
            "robot.options.kinematics_factory must use module:function syntax"
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        if not callable(factory):
            raise TypeError("declared engine factory is not callable")
        engine = factory(robot_config)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"failed to load kinematics engine {qualified_name!r}: {exc}"
        ) from exc

    required = ("encode_policy_state", "policy_chunk_to_robot_targets")
    missing = [name for name in required if not callable(getattr(engine, name, None))]
    if missing:
        raise ConfigurationError(
            "kinematics engine is missing callable(s): " + ", ".join(missing)
        )
    return engine, qualified_name


class LiberoCartesianKinematicsAdapter:
    """Adapt a LIBERO-style Cartesian policy through a supplied IK/FK engine."""

    rtc_native_safe = False

    def __init__(
        self, robot_config: RobotConfig, engine: Any, engine_name: str
    ) -> None:
        robot_config.validate()
        self.robot_config = robot_config
        self.config = robot_config.action_adapter
        self.engine = engine
        self.engine_name = engine_name
        self.robot_dim = robot_config.action_dim
        self.policy_dim = self.config.resolved_policy_dim(self.robot_dim)
        self.policy_state_dim = self.config.resolved_policy_state_dim(self.robot_dim)

        if self.config.policy_frame != "end_effector":
            raise ConfigurationError(
                "LIBERO Cartesian adapter requires policy_frame='end_effector'"
            )
        if self.config.policy_dim <= 0 or self.config.policy_state_dim <= 0:
            raise ConfigurationError(
                "kinematics adapter requires explicit policy_dim and policy_state_dim"
            )
        if len(self.config.policy_names) != self.policy_dim:
            raise ConfigurationError(
                "kinematics adapter requires explicit policy_names for every action"
            )
        if len(self.config.policy_state_names) != self.policy_state_dim:
            raise ConfigurationError(
                "kinematics adapter requires explicit policy_state_names"
            )

        self.robot_joint_names = tuple(robot_config.joint_names)
        self.policy_state_names = tuple(self.config.policy_state_names)
        self.lower = _finite_f32(robot_config.joint_lower, name="robot joint lower")
        self.upper = _finite_f32(robot_config.joint_upper, name="robot joint upper")
        self.max_step = _finite_f32(robot_config.max_step, name="robot max_step")
        self.expected_period_s = 1.0 / float(robot_config.control_hz)

    def _validate_required_cameras(self, observation: Observation) -> None:
        missing = sorted(set(self.config.required_cameras) - set(observation.images))
        if missing:
            raise ShapeMismatchError(
                "observation is missing required policy cameras: " + ", ".join(missing)
            )

    def _validate_observation(self, observation: Observation) -> NDArray[np.float32]:
        self._validate_required_cameras(observation)
        _positive_timestamp(observation.timestamp_ns, name="observation timestamp_ns")
        _positive_timestamp(
            observation.state.timestamp_ns, name="robot state timestamp_ns"
        )
        if tuple(observation.state.joint_names) != self.robot_joint_names:
            raise ShapeMismatchError(
                "robot state joint_names do not match configured robot order"
            )
        positions = _finite_f32(
            observation.state.joint_positions, name="observed robot positions"
        )
        if positions.shape != (self.robot_dim,):
            raise ShapeMismatchError(
                f"observed robot positions must have shape ({self.robot_dim},)"
            )
        if np.any(positions < self.lower) or np.any(positions > self.upper):
            raise SafetyViolationError(
                "observed robot state is outside configured joint limits"
            )
        return positions

    def _normalize_policy_state(
        self, value: Any, observation: Observation
    ) -> RobotState:
        if isinstance(value, RobotState):
            state = value
            if tuple(state.joint_names) != self.policy_state_names:
                raise ShapeMismatchError(
                    "kinematics policy-state names do not match policy_state_names"
                )
            if state.timestamp_ns != observation.state.timestamp_ns:
                raise ShapeMismatchError(
                    "kinematics policy state must preserve robot-state timestamp"
                )
            positions = _finite_f32(
                state.joint_positions, name="kinematics policy state"
            )
            if positions.shape != (self.policy_state_dim,):
                raise ShapeMismatchError(
                    "kinematics policy state has the wrong dimension"
                )
            return state

        positions = _finite_f32(value, name="kinematics policy state")
        if positions.shape != (self.policy_state_dim,):
            raise ShapeMismatchError(
                f"kinematics policy state must have shape ({self.policy_state_dim},)"
            )
        gripper_position = None
        gripper_indices = [
            index
            for index, name in enumerate(self.policy_state_names)
            if "gripper" in name.lower()
        ]
        if len(gripper_indices) == 1:
            gripper_position = float(positions[gripper_indices[0]])
        return RobotState(
            joint_positions=positions,
            joint_names=self.policy_state_names,
            timestamp_ns=observation.state.timestamp_ns,
            gripper_position=gripper_position,
            fault=observation.state.fault,
        )

    def observation_to_policy_state(self, observation: Observation) -> RobotState:
        """Encode one validated robot observation into checkpoint state space."""

        self._validate_observation(observation)
        try:
            encoded = self.engine.encode_policy_state(observation)
        except Exception as exc:
            raise ShapeMismatchError(
                f"kinematics encode_policy_state failed: {exc}"
            ) from exc
        return self._normalize_policy_state(encoded, observation)

    def _validate_chunk_provenance(
        self, chunk: ActionChunk, observation: Observation
    ) -> NDArray[np.float32]:
        if not math.isfinite(chunk.period_s) or chunk.period_s <= 0:
            raise ShapeMismatchError("action period_s must be finite and positive")
        if not math.isclose(
            chunk.period_s,
            self.expected_period_s,
            rel_tol=_PERIOD_REL_TOLERANCE,
            abs_tol=_PERIOD_ABS_TOLERANCE,
        ):
            raise ShapeMismatchError(
                f"action period_s={chunk.period_s} does not match configured "
                f"control period {self.expected_period_s}"
            )
        source_ns = _positive_timestamp(
            chunk.source_observation_ns, name="action source_observation_ns"
        )
        generated_ns = _positive_timestamp(
            chunk.generated_ns, name="action generated_ns"
        )
        if source_ns != observation.timestamp_ns:
            raise ShapeMismatchError(
                "action chunk is not anchored to the supplied observation"
            )
        if generated_ns < source_ns:
            raise ShapeMismatchError(
                "action generated_ns predates its source observation"
            )
        values = _finite_f32(chunk.values, name="policy action chunk")
        if values.ndim != 2 or values.shape[1] != self.policy_dim:
            raise ShapeMismatchError(
                f"policy action chunk must have shape [T, {self.policy_dim}]"
            )
        return values

    def _validate_targets(
        self,
        value: Any,
        *,
        horizon: int,
        observed_positions: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        targets = _finite_f32(value, name="kinematics robot targets")
        if targets.shape != (horizon, self.robot_dim):
            raise ShapeMismatchError(
                "kinematics engine must return absolute targets with shape "
                f"({horizon}, {self.robot_dim})"
            )
        outside = np.logical_or(
            targets < self.lower[None, :] - _LIMIT_TOLERANCE,
            targets > self.upper[None, :] + _LIMIT_TOLERANCE,
        )
        if np.any(outside):
            raise SafetyViolationError(
                "kinematics target exceeds configured robot joint limits"
            )
        previous = np.concatenate((observed_positions[None, :], targets[:-1]), axis=0)
        if np.any(
            np.abs(targets - previous) > self.max_step[None, :] + _LIMIT_TOLERANCE
        ):
            raise SafetyViolationError(
                "kinematics target exceeds per-command robot max_step"
            )
        return targets

    def adapt_action_chunk(
        self, chunk: ActionChunk, observation: Observation
    ) -> ActionChunk:
        """Convert a checkpoint action chunk to checked absolute joint targets."""

        if not self.robot_config.dry_run and not self.config.validated:
            raise PermissionError("real motion requires action_adapter.validated=true")
        observed_positions = self._validate_observation(observation)
        policy_values = self._validate_chunk_provenance(chunk, observation)
        try:
            raw_targets = self.engine.policy_chunk_to_robot_targets(
                observation,
                policy_values.copy(),
                chunk.period_s,
                self.config,
            )
        except (SafetyViolationError, ShapeMismatchError):
            raise
        except Exception as exc:
            raise ShapeMismatchError(
                f"kinematics policy_chunk_to_robot_targets failed: {exc}"
            ) from exc
        targets = self._validate_targets(
            raw_targets,
            horizon=chunk.horizon,
            observed_positions=observed_positions,
        )
        metadata = dict(chunk.metadata)
        metadata["libero_cartesian_kinematics"] = {
            "engine_factory": self.engine_name,
            "policy_frame": self.config.policy_frame,
            "output_semantics": "absolute_robot_joint_target",
            "calibration_validated": self.config.validated,
            "calibration_id": self.config.calibration_id,
            "rtc_native_safe": False,
        }
        return ActionChunk(
            values=targets,
            period_s=chunk.period_s,
            source_observation_ns=chunk.source_observation_ns,
            generated_ns=chunk.generated_ns,
            model_values=None,
            committed_prefix=chunk.committed_prefix,
            metadata=metadata,
        )


def create_libero_cartesian_adapter(
    robot_config: RobotConfig,
) -> LiberoCartesianKinematicsAdapter:
    """Load the configured deployment engine and construct the safe adapter."""

    robot_config.validate()
    engine, engine_name = _load_engine(robot_config)
    return LiberoCartesianKinematicsAdapter(robot_config, engine, engine_name)
