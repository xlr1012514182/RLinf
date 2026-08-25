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

"""Fail-closed RoboTwin dual-Aloha to single-arm retargeting boundary.

The source-locked RLinf pi0.5 RoboTwin policy consumes a 14-dimensional
dual-Aloha state, uses three cameras, and emits an H=50 chunk of absolute
dual-Aloha joint/gripper targets after the OpenPI output transforms.  None of
those coordinates has a safe generic correspondence to an SO101 or Dobot Nova.
This module therefore validates the checkpoint contract and delegates the two
deployment-specific mappings to a user-supplied engine::

    encode_policy_state(observation) -> array[14] | RobotState
    policy_chunk_to_robot_targets(
        observation, policy_values, period_s, action_adapter_config
    ) -> array[50, robot_dim]

Set ``robot.options.robotwin_retargeting_factory`` to the reviewed
``module:function`` that constructs that engine from ``RobotConfig``.  The
checked-in example values are deliberately locked placeholders and are
rejected.  For non-dry motion the engine must additionally declare
``validated = True`` and a ``calibration_id`` exactly matching the action
adapter configuration.  Outputs are never clipped.

Native RTC is disabled: a general dual-arm-to-single-arm retargeting map need
not be invertible, so robot-space overlap constraints cannot safely be mapped
back into the checkpoint's denoising coordinates.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import ActionChunk, Observation, RobotState
from ..errors import ConfigurationError, SafetyViolationError, ShapeMismatchError

ROBOTWIN_PI05_HORIZON = 50
ROBOTWIN_PI05_DIM = 14
ROBOTWIN_PI05_CAMERAS = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)
ROBOTWIN_PI05_JOINTS = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
)

_SOURCE_UNITS = (
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "normalized",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "normalized",
)
_PERIOD_REL_TOLERANCE = 1e-6
_PERIOD_ABS_TOLERANCE = 1e-9
_LIMIT_TOLERANCE = 1e-6
_PLACEHOLDER_MARKERS = (
    "REPLACE",
    "PLACEHOLDER",
    "TODO",
    "MODULE:FUNCTION",
    "LOCKED",
    "YOUR_",
)


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


def _load_retargeting_engine(robot_config: RobotConfig) -> tuple[Any, str]:
    value = robot_config.options.get("robotwin_retargeting_factory")
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            "robot.options.robotwin_retargeting_factory must name a reviewed "
            "module:function"
        )
    qualified_name = value.strip()
    if any(marker in qualified_name.upper() for marker in _PLACEHOLDER_MARKERS):
        raise ConfigurationError(
            "robot.options.robotwin_retargeting_factory is still a locked placeholder"
        )
    module_name, separator, factory_name = qualified_name.partition(":")
    if not separator or not module_name.strip() or not factory_name.strip():
        raise ConfigurationError(
            "robot.options.robotwin_retargeting_factory must use module:function syntax"
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        if not callable(factory):
            raise TypeError("declared retargeting engine factory is not callable")
        engine = factory(robot_config)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"failed to load RoboTwin retargeting engine {qualified_name!r}: {exc}"
        ) from exc

    required = ("encode_policy_state", "policy_chunk_to_robot_targets")
    missing = [name for name in required if not callable(getattr(engine, name, None))]
    if missing:
        raise ConfigurationError(
            "RoboTwin retargeting engine is missing callable(s): " + ", ".join(missing)
        )
    return engine, qualified_name


class RoboTwinPi05H50SingleArmRetargetingAdapter:
    """Validate and retarget one source-locked RoboTwin pi0.5 contract."""

    rtc_native_safe = False

    def __init__(
        self,
        robot_config: RobotConfig,
        engine: Any,
        engine_factory: str,
    ) -> None:
        robot_config.validate()
        self.robot_config = robot_config
        self.config = robot_config.action_adapter
        self.engine = engine
        self.engine_factory = engine_factory
        self.robot_dim = robot_config.action_dim
        self.robot_joint_names = tuple(robot_config.joint_names)
        self.lower = _finite_f32(robot_config.joint_lower, name="robot joint lower")
        self.upper = _finite_f32(robot_config.joint_upper, name="robot joint upper")
        self.max_step = _finite_f32(robot_config.max_step, name="robot max_step")
        self.expected_period_s = 1.0 / float(robot_config.control_hz)
        self.maximum_skew_ns = int(
            float(robot_config.maximum_state_camera_skew_ms) * 1_000_000
        )
        self._validate_static_contract()

    def _validate_static_contract(self) -> None:
        expected_robot = {"so101": 6, "dobot_nova": 7}
        expected_dim = expected_robot.get(self.robot_config.backend)
        if expected_dim is None or self.robot_dim != expected_dim:
            raise ConfigurationError(
                "RoboTwin pi0.5 retargeting supports only a six-dimensional "
                "SO101 or seven-dimensional Dobot Nova target"
            )
        if self.config.policy_dim != ROBOTWIN_PI05_DIM:
            raise ConfigurationError("RoboTwin pi0.5 policy_dim must be exactly 14")
        if self.config.policy_state_dim != ROBOTWIN_PI05_DIM:
            raise ConfigurationError(
                "RoboTwin pi0.5 policy_state_dim must be exactly 14"
            )
        if tuple(self.config.policy_names) != ROBOTWIN_PI05_JOINTS:
            raise ConfigurationError(
                "RoboTwin pi0.5 policy_names must use the source-locked "
                "dual-Aloha order"
            )
        if tuple(self.config.policy_state_names) != ROBOTWIN_PI05_JOINTS:
            raise ConfigurationError(
                "RoboTwin pi0.5 policy_state_names must use the source-locked "
                "dual-Aloha order"
            )
        if tuple(self.config.policy_units) != _SOURCE_UNITS:
            raise ConfigurationError(
                "RoboTwin pi0.5 policy_units do not match the dual-Aloha contract"
            )
        if (
            self.config.policy_frame != "joint"
            or self.config.policy_state_frame != "joint"
        ):
            raise ConfigurationError(
                "RoboTwin pi0.5 action and state frames must both be 'joint'"
            )
        if (
            self.config.action_mode != "absolute"
            or tuple(self.config.coordinate_modes) != ("absolute",) * ROBOTWIN_PI05_DIM
        ):
            raise ConfigurationError(
                "RoboTwin pi0.5 environment outputs must be declared as 14 "
                "absolute coordinates"
            )
        if tuple(self.config.required_cameras) != ROBOTWIN_PI05_CAMERAS:
            raise ConfigurationError(
                "RoboTwin pi0.5 requires cam_high, cam_left_wrist, and "
                "cam_right_wrist in source order"
            )
        if self.config.robot_from_policy or self.config.scale or self.config.offset:
            raise ConfigurationError(
                "RoboTwin single-arm retargeting must not declare an affine "
                "policy-to-robot calibration"
            )

    def _validate_observation(self, observation: Observation) -> NDArray[np.float32]:
        observation_ns = _positive_timestamp(
            observation.timestamp_ns, name="observation timestamp_ns"
        )
        state_ns = _positive_timestamp(
            observation.state.timestamp_ns, name="robot state timestamp_ns"
        )
        if state_ns > observation_ns:
            raise ShapeMismatchError(
                "robot state timestamp cannot postdate the synchronized observation"
            )
        missing = sorted(set(ROBOTWIN_PI05_CAMERAS) - set(observation.images))
        if missing:
            raise ShapeMismatchError(
                "observation is missing required RoboTwin cameras: "
                + ", ".join(missing)
            )
        raw_timestamps = observation.metadata.get("camera_timestamps_ns")
        if not isinstance(raw_timestamps, Mapping):
            raise ShapeMismatchError(
                "observation metadata must include camera_timestamps_ns"
            )
        missing_timestamps = sorted(set(ROBOTWIN_PI05_CAMERAS) - set(raw_timestamps))
        if missing_timestamps:
            raise ShapeMismatchError(
                "camera_timestamps_ns is missing RoboTwin cameras: "
                + ", ".join(missing_timestamps)
            )
        camera_timestamps = tuple(
            _positive_timestamp(
                raw_timestamps[name], name=f"camera {name} timestamp_ns"
            )
            for name in ROBOTWIN_PI05_CAMERAS
        )
        if observation_ns != max(camera_timestamps):
            raise ShapeMismatchError(
                "observation timestamp must equal the latest required camera timestamp"
            )
        skew_ns = max(camera_timestamps) - min(camera_timestamps)
        state_skew_ns = max(abs(state_ns - value) for value in camera_timestamps)
        if max(skew_ns, state_skew_ns) > self.maximum_skew_ns:
            raise ShapeMismatchError(
                "RoboTwin state/camera timestamps exceed maximum_state_camera_skew_ms"
            )
        if tuple(observation.state.joint_names) != self.robot_joint_names:
            raise ShapeMismatchError(
                "robot state joint_names do not match configured target-arm order"
            )
        positions = _finite_f32(
            observation.state.joint_positions, name="observed robot positions"
        )
        if positions.shape != (self.robot_dim,):
            raise ShapeMismatchError(
                f"observed robot positions must have shape ({self.robot_dim},)"
            )
        if observation.state.fault:
            raise SafetyViolationError(
                f"target robot reports a fault: {observation.state.fault}"
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
            if tuple(value.joint_names) != ROBOTWIN_PI05_JOINTS:
                raise ShapeMismatchError(
                    "retargeting policy-state names do not match dual-Aloha order"
                )
            if value.timestamp_ns != observation.state.timestamp_ns:
                raise ShapeMismatchError(
                    "retargeting policy state must preserve robot-state timestamp"
                )
            if value.fault != observation.state.fault:
                raise ShapeMismatchError(
                    "retargeting policy state must preserve fault provenance"
                )
            positions = _finite_f32(
                value.joint_positions, name="retargeted RoboTwin policy state"
            )
            if positions.shape != (ROBOTWIN_PI05_DIM,):
                raise ShapeMismatchError(
                    "retargeted RoboTwin policy state must have shape (14,)"
                )
            return value

        positions = _finite_f32(value, name="retargeted RoboTwin policy state")
        if positions.shape != (ROBOTWIN_PI05_DIM,):
            raise ShapeMismatchError(
                "retargeted RoboTwin policy state must have shape (14,)"
            )
        return RobotState(
            joint_positions=positions,
            joint_names=ROBOTWIN_PI05_JOINTS,
            timestamp_ns=observation.state.timestamp_ns,
            fault=observation.state.fault,
        )

    def observation_to_policy_state(self, observation: Observation) -> RobotState:
        """Encode a synchronized single-arm observation as dual-Aloha state."""

        self._validate_observation(observation)
        try:
            encoded = self.engine.encode_policy_state(observation)
        except Exception as exc:
            raise ShapeMismatchError(
                f"RoboTwin encode_policy_state failed: {exc}"
            ) from exc
        return self._normalize_policy_state(encoded, observation)

    def _validate_chunk(
        self, chunk: ActionChunk, observation: Observation
    ) -> NDArray[np.float32]:
        if chunk.horizon != ROBOTWIN_PI05_HORIZON:
            raise ShapeMismatchError("RoboTwin pi0.5 action horizon must be exactly 50")
        if chunk.action_dim != ROBOTWIN_PI05_DIM:
            raise ShapeMismatchError(
                "RoboTwin pi0.5 action chunk must have shape [50, 14]"
            )
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
        return _finite_f32(chunk.values, name="RoboTwin policy action chunk")

    def _validate_real_motion_evidence(self) -> None:
        if self.robot_config.dry_run:
            return
        if not self.config.validated:
            raise PermissionError("real motion requires action_adapter.validated=true")
        if any(
            marker in self.config.calibration_id.upper()
            for marker in _PLACEHOLDER_MARKERS
        ):
            raise PermissionError(
                "real motion requires a non-placeholder retargeting calibration_id"
            )
        if getattr(self.engine, "validated", None) is not True:
            raise PermissionError(
                "real motion requires retargeting engine validated=True"
            )
        engine_calibration_id = getattr(self.engine, "calibration_id", None)
        if (
            not isinstance(engine_calibration_id, str)
            or engine_calibration_id != self.config.calibration_id
        ):
            raise PermissionError(
                "retargeting engine calibration_id must exactly match the "
                "action adapter calibration_id"
            )

    def _validate_targets(
        self,
        value: Any,
        observed_positions: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        targets = _finite_f32(value, name="retargeted absolute robot targets")
        expected_shape = (ROBOTWIN_PI05_HORIZON, self.robot_dim)
        if targets.shape != expected_shape:
            raise ShapeMismatchError(
                "retargeting engine must return absolute targets with shape "
                f"{expected_shape}"
            )
        if np.any(targets < self.lower[None, :] - _LIMIT_TOLERANCE) or np.any(
            targets > self.upper[None, :] + _LIMIT_TOLERANCE
        ):
            raise SafetyViolationError(
                "retargeted target exceeds configured robot joint limits"
            )
        previous = np.concatenate((observed_positions[None, :], targets[:-1]), axis=0)
        if np.any(
            np.abs(targets - previous) > self.max_step[None, :] + _LIMIT_TOLERANCE
        ):
            raise SafetyViolationError(
                "retargeted target exceeds per-command robot max_step"
            )
        return targets

    def adapt_action_chunk(
        self, chunk: ActionChunk, observation: Observation
    ) -> ActionChunk:
        """Convert one H=50 dual-Aloha chunk into checked absolute targets."""

        self._validate_real_motion_evidence()
        observed_positions = self._validate_observation(observation)
        policy_values = self._validate_chunk(chunk, observation)
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
                f"RoboTwin policy_chunk_to_robot_targets failed: {exc}"
            ) from exc
        targets = self._validate_targets(raw_targets, observed_positions)
        metadata = dict(chunk.metadata)
        metadata["robotwin_pi05_h50_retargeting"] = {
            "engine_factory": self.engine_factory,
            "source_contract": "robotwin_dual_aloha_absolute_h50_a14",
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


def create_robotwin_pi05_h50_retargeting_adapter(
    robot_config: RobotConfig,
) -> RoboTwinPi05H50SingleArmRetargetingAdapter:
    """Load a reviewed engine and construct the fail-closed adapter."""

    robot_config.validate()
    engine, engine_factory = _load_retargeting_engine(robot_config)
    return RoboTwinPi05H50SingleArmRetargetingAdapter(
        robot_config,
        engine,
        engine_factory,
    )
