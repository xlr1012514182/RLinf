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

"""Validated configuration dataclasses for the Fibocom VLA stack."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError


@dataclass(frozen=True)
class ResidualRLConfig:
    """Frozen-base residual policy and hybrid PPO/twin-Q settings."""

    enabled: bool = True
    feature_dim: int = 2048
    action_horizon: int = 50
    action_dim: int = 7
    actor_hidden_dim: int = 562
    actor_bottleneck_dim: int = 512
    max_residual: float = 0.08
    action_low: float = -1.0
    action_high: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 1
    entropy_coefficient: float = 0.0
    anchor_coefficient: float = 0.02
    q_policy_coefficient: float = 0.05
    critic_coefficient: float = 0.5
    value_coefficient: float = 0.5
    target_tau: float = 0.005
    second_failure_penalty: float = -1.0
    target_actor_parameters: int = 1_820_000
    parameter_tolerance: int = 10_000

    def validate(self) -> None:
        """Validate shape, probability, and parameter-budget settings."""

        if not isinstance(self.enabled, bool):
            raise ConfigurationError("residual_rl.enabled must be boolean")
        positive = {
            "feature_dim": self.feature_dim,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "actor_hidden_dim": self.actor_hidden_dim,
            "actor_bottleneck_dim": self.actor_bottleneck_dim,
            "max_residual": self.max_residual,
            "target_actor_parameters": self.target_actor_parameters,
            "parameter_tolerance": self.parameter_tolerance,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if (
            not math.isfinite(self.action_low)
            or not math.isfinite(self.action_high)
            or self.action_low >= self.action_high
        ):
            raise ConfigurationError("action_low must be below action_high")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ConfigurationError("gamma and gae_lambda must be in [0, 1]")
        if not 0 < self.ppo_clip_ratio < 1:
            raise ConfigurationError("ppo_clip_ratio must be in (0, 1)")
        if self.ppo_epochs != 1:
            raise ConfigurationError(
                "This reproduction fixes ppo_epochs=1 to match the declared single-round update"
            )
        coefficients = {
            "entropy_coefficient": self.entropy_coefficient,
            "anchor_coefficient": self.anchor_coefficient,
            "q_policy_coefficient": self.q_policy_coefficient,
            "critic_coefficient": self.critic_coefficient,
            "value_coefficient": self.value_coefficient,
        }
        for name, value in coefficients.items():
            if not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be non-negative, got {value}")
        if not 0 < self.target_tau <= 1:
            raise ConfigurationError("target_tau must be in (0, 1]")
        if (
            not math.isfinite(self.second_failure_penalty)
            or self.second_failure_penalty >= 0
        ):
            raise ConfigurationError(
                "second_failure_penalty must be finite and negative"
            )


@dataclass(frozen=True)
class SpeculativeConfig:
    """Continuous-action draft/parallel-verification settings."""

    enabled: bool = True
    relative_error_threshold: float = 0.12
    absolute_error_threshold: float = 0.03
    minimum_prefix: int = 2
    fallback_acceptance_ratio: float = 0.25
    fallback_cooldown_chunks: int = 3

    def validate(self) -> None:
        """Validate acceptance and fallback thresholds."""

        thresholds = (
            self.relative_error_threshold,
            self.absolute_error_threshold,
        )
        if any(not math.isfinite(value) or value < 0 for value in thresholds):
            raise ConfigurationError("verification thresholds must be non-negative")
        if self.minimum_prefix < 0:
            raise ConfigurationError("minimum_prefix must be non-negative")
        if not 0 <= self.fallback_acceptance_ratio <= 1:
            raise ConfigurationError("fallback_acceptance_ratio must be in [0, 1]")
        if (
            isinstance(self.fallback_cooldown_chunks, bool)
            or not isinstance(self.fallback_cooldown_chunks, Integral)
            or self.fallback_cooldown_chunks < 0
        ):
            raise ConfigurationError(
                "fallback_cooldown_chunks must be a non-negative integer"
            )


@dataclass(frozen=True)
class RTCConfig:
    """Real-time chunking settings."""

    enabled: bool = True
    execution_horizon: int = 8
    queue_threshold: int = 2
    overlap_horizon: int = 16
    guidance_weight: float = 1.0
    guidance_decay: float = 0.82
    planner_timeout_s: float = 5.0
    planner_shutdown_timeout_s: float = 5.0

    def validate(self, action_horizon: int) -> None:
        """Validate overlap and execution horizons."""

        if not 0 < self.execution_horizon <= action_horizon:
            raise ConfigurationError("execution_horizon must be in [1, action_horizon]")
        if not 0 <= self.overlap_horizon <= action_horizon:
            raise ConfigurationError("overlap_horizon exceeds action_horizon")
        if not 0 < self.guidance_decay <= 1:
            raise ConfigurationError("guidance_decay must be in (0, 1]")
        if not math.isfinite(self.guidance_weight) or self.guidance_weight < 0:
            raise ConfigurationError("guidance_weight must be finite and non-negative")
        timeouts = (self.planner_timeout_s, self.planner_shutdown_timeout_s)
        if self.queue_threshold < 0 or any(
            not math.isfinite(value) or value <= 0 for value in timeouts
        ):
            raise ConfigurationError("RTC queue/timeout settings are invalid")


@dataclass(frozen=True)
class ActionAdapterConfig:
    """Declared policy/robot semantics and affine calibration evidence.

    ``robot_from_policy[r]`` selects the policy coordinate that corresponds to
    robot joint ``r``. For the built-in adapter this must be a permutation;
    ``robot = policy[..., robot_from_policy] * scale + offset``. Empty vectors
    resolve to identity only for unvalidated dry runs.

    The generic implementation intentionally supports only ``joint`` policy
    action/state frames. Other frames may be declared in placeholder configs,
    but cannot be marked validated or used for real motion without a separate,
    explicitly reviewed kinematics adapter.
    """

    implementation: str = "generic_affine_joint"
    action_mode: str = "absolute"
    coordinate_modes: tuple[str, ...] = ()
    policy_frame: str = "joint"
    policy_dim: int = 0
    policy_names: tuple[str, ...] = ()
    policy_units: tuple[str, ...] = ()
    policy_state_frame: str = "joint"
    policy_state_dim: int = 0
    policy_state_names: tuple[str, ...] = ()
    required_cameras: tuple[str, ...] = ()
    robot_units: tuple[str, ...] = ()
    robot_from_policy: tuple[int, ...] = ()
    scale: tuple[float, ...] = ()
    offset: tuple[float, ...] = ()
    validated: bool = False
    calibration_id: str = ""

    def __post_init__(self) -> None:
        """Normalize JSON lists into immutable tuples."""

        tuple_fields = (
            "coordinate_modes",
            "policy_names",
            "policy_units",
            "policy_state_names",
            "required_cameras",
            "robot_units",
            "robot_from_policy",
            "scale",
            "offset",
        )
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def resolved_policy_dim(self, robot_dim: int) -> int:
        """Return the explicit policy dimension or the dry-run identity default."""

        return self.policy_dim or robot_dim

    def resolved_policy_state_dim(self, robot_dim: int) -> int:
        """Return the declared policy-state dimension."""

        return self.policy_state_dim or self.resolved_policy_dim(robot_dim)

    def validate(
        self,
        *,
        robot_dim: int,
        robot_joint_names: tuple[str, ...],
        dry_run: bool,
    ) -> None:
        """Validate the declaration and enforce the real-motion gate."""

        modes = {
            "absolute",
            "delta_from_observation",
            "integrated_delta",
            "velocity",
        }
        if self.action_mode not in modes:
            raise ConfigurationError(
                f"unsupported policy action_mode: {self.action_mode}"
            )
        frames = {"joint", "end_effector", "checkpoint_native"}
        if self.policy_frame not in frames or self.policy_state_frame not in frames:
            raise ConfigurationError("unsupported policy action/state frame")
        dimensions = (self.policy_dim, self.policy_state_dim)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in dimensions
        ):
            raise ConfigurationError("policy dimensions must be integers")
        if self.policy_dim < 0 or self.policy_state_dim < 0:
            raise ConfigurationError("policy dimensions must be non-negative")
        policy_dim = self.resolved_policy_dim(robot_dim)
        state_dim = self.resolved_policy_state_dim(robot_dim)
        if policy_dim <= 0 or state_dim <= 0:
            raise ConfigurationError("resolved policy dimensions must be positive")
        if self.coordinate_modes:
            if len(self.coordinate_modes) != policy_dim:
                raise ConfigurationError(
                    f"action_adapter.coordinate_modes must have length {policy_dim}"
                )
            invalid_modes = sorted(set(self.coordinate_modes) - modes)
            if invalid_modes:
                raise ConfigurationError(
                    "unsupported coordinate action modes: " + ", ".join(invalid_modes)
                )

        sized_fields = {
            "policy_names": (self.policy_names, policy_dim),
            "policy_units": (self.policy_units, policy_dim),
            "policy_state_names": (self.policy_state_names, state_dim),
            "robot_units": (self.robot_units, robot_dim),
            "robot_from_policy": (self.robot_from_policy, robot_dim),
            "scale": (self.scale, robot_dim),
            "offset": (self.offset, robot_dim),
        }
        for name, (values, expected) in sized_fields.items():
            if values and len(values) != expected:
                raise ConfigurationError(
                    f"action_adapter.{name} must have length {expected}"
                )
        for name, values in {
            "policy_names": self.policy_names,
            "policy_units": self.policy_units,
            "policy_state_names": self.policy_state_names,
            "required_cameras": self.required_cameras,
            "robot_units": self.robot_units,
        }.items():
            if any(not str(value).strip() for value in values):
                raise ConfigurationError(f"action_adapter.{name} contains blanks")
        for name, values in {
            "policy_names": self.policy_names,
            "policy_state_names": self.policy_state_names,
        }.items():
            if len(set(values)) != len(values):
                raise ConfigurationError(f"action_adapter.{name} must be unique")
        if len(set(self.required_cameras)) != len(self.required_cameras):
            raise ConfigurationError("required_cameras must be unique")
        if self.robot_from_policy:
            if any(
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or not 0 <= index < policy_dim
                for index in self.robot_from_policy
            ):
                raise ConfigurationError("robot_from_policy contains invalid indices")
            if len(set(self.robot_from_policy)) != len(self.robot_from_policy):
                raise ConfigurationError("robot_from_policy indices must be unique")
        if self.scale and (
            any(not math.isfinite(value) for value in self.scale)
            or any(value == 0 for value in self.scale)
        ):
            raise ConfigurationError("action_adapter.scale must be finite and non-zero")
        if self.offset and any(not math.isfinite(value) for value in self.offset):
            raise ConfigurationError("action_adapter.offset must be finite")

        generic_implementation = self.implementation == "generic_affine_joint"
        if not generic_implementation:
            module_name, separator, attribute = self.implementation.partition(":")
            if not separator or not module_name or not attribute:
                raise ConfigurationError(
                    "custom action_adapter.implementation must use module:function"
                )
        if self.validated:
            required = {
                "policy_dim": self.policy_dim,
                "policy_names": self.policy_names,
                "policy_units": self.policy_units,
                "policy_state_dim": self.policy_state_dim,
                "policy_state_names": self.policy_state_names,
                "robot_units": self.robot_units,
                "calibration_id": self.calibration_id.strip(),
            }
            if generic_implementation:
                required.update(
                    {
                        "robot_from_policy": self.robot_from_policy,
                        "scale": self.scale,
                        "offset": self.offset,
                    }
                )
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ConfigurationError(
                    "validated calibration is missing explicit fields: "
                    + ", ".join(missing)
                )
            if "REPLACE" in self.calibration_id.upper():
                raise ConfigurationError("calibration_id is still a placeholder")
            if generic_implementation and (
                self.policy_frame != "joint" or self.policy_state_frame != "joint"
            ):
                raise ConfigurationError(
                    "validated generic adapter requires joint action/state frames; "
                    "end-effector/checkpoint-native spaces require reviewed IK/FK"
                )
            if generic_implementation and (
                policy_dim != robot_dim or state_dim != policy_dim
            ):
                raise ConfigurationError(
                    "validated generic adapter requires equal policy/state/robot dimensions"
                )
            if generic_implementation and sorted(self.robot_from_policy) != list(
                range(policy_dim)
            ):
                raise ConfigurationError(
                    "validated robot_from_policy must be a complete permutation"
                )
            if generic_implementation and tuple(self.policy_state_names) != tuple(
                self.policy_names
            ):
                raise ConfigurationError(
                    "joint policy_state_names must match policy_names for inversion"
                )
            if len(robot_joint_names) != robot_dim:
                raise ConfigurationError("robot joint-name declaration is inconsistent")
        if not dry_run and not self.validated:
            raise ConfigurationError(
                "real motion requires action_adapter.validated=true"
            )


@dataclass(frozen=True)
class RobotConfig:
    """Backend-neutral robot configuration."""

    backend: str = "mock"
    action_dim: int = 7
    control_hz: float = 20.0
    maximum_source_age_s: float = 1.0
    maximum_generation_age_s: float = 0.75
    maximum_observation_age_s: float = 0.5
    maximum_state_camera_skew_ms: float = 75.0
    joint_names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
        "aux",
    )
    joint_lower: tuple[float, ...] = (-180.0,) * 7
    joint_upper: tuple[float, ...] = (180.0,) * 7
    max_step: tuple[float, ...] = (5.0,) * 7
    dry_run: bool = True
    action_adapter: ActionAdapterConfig = field(default_factory=ActionAdapterConfig)
    options: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate dimensions and safety envelopes."""

        if self.backend not in {"mock", "so101", "dobot_nova", "ros2_joint"}:
            raise ConfigurationError(f"unsupported robot backend: {self.backend}")
        if (
            isinstance(self.action_dim, bool)
            or not isinstance(self.action_dim, Integral)
            or self.action_dim <= 0
            or not math.isfinite(self.control_hz)
            or self.control_hz <= 0
        ):
            raise ConfigurationError("action_dim and control_hz must be positive")
        freshness_limits = (
            self.maximum_source_age_s,
            self.maximum_generation_age_s,
            self.maximum_observation_age_s,
            self.maximum_state_camera_skew_ms,
        )
        if any(not math.isfinite(value) or value <= 0 for value in freshness_limits):
            raise ConfigurationError(
                "action/observation age and state-camera skew gates must be "
                "finite and positive"
            )
        vectors = (self.joint_names, self.joint_lower, self.joint_upper, self.max_step)
        if any(len(value) != self.action_dim for value in vectors):
            raise ConfigurationError(
                "joint_names, limits, and max_step must match action_dim"
            )
        if any(low >= high for low, high in zip(self.joint_lower, self.joint_upper)):
            raise ConfigurationError("each joint lower limit must be below upper")
        if any(step <= 0 for step in self.max_step):
            raise ConfigurationError("max_step entries must be positive")
        if any(not name.strip() for name in self.joint_names):
            raise ConfigurationError("joint_names must not contain blanks")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ConfigurationError("joint_names must be unique")
        numeric_vectors = (self.joint_lower, self.joint_upper, self.max_step)
        if any(
            not math.isfinite(value) for vector in numeric_vectors for value in vector
        ):
            raise ConfigurationError("joint limits and max_step must be finite")
        self.action_adapter.validate(
            robot_dim=self.action_dim,
            robot_joint_names=self.joint_names,
            dry_run=self.dry_run,
        )


@dataclass(frozen=True)
class CameraConfig:
    """One camera backend configuration."""

    name: str = "main"
    backend: str = "mock"
    width: int = 640
    height: int = 480
    fps: int = 30
    options: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate camera backend and frame geometry."""

        if self.backend not in {"mock", "opencv", "realsense", "ros2_image"}:
            raise ConfigurationError(f"unsupported camera backend: {self.backend}")
        if not self.name or min(self.width, self.height, self.fps) <= 0:
            raise ConfigurationError("camera name and dimensions must be valid")


@dataclass(frozen=True)
class StackConfig:
    """Top-level configuration."""

    residual_rl: ResidualRLConfig = field(default_factory=ResidualRLConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    rtc: RTCConfig = field(default_factory=RTCConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    cameras: tuple[CameraConfig, ...] = field(default_factory=lambda: (CameraConfig(),))

    def validate(self) -> None:
        """Validate all nested configuration and cross-field invariants."""

        self.residual_rl.validate()
        self.speculative.validate()
        self.rtc.validate(self.residual_rl.action_horizon)
        self.robot.validate()
        for camera in self.cameras:
            camera.validate()
        policy_action_dim = self.robot.action_adapter.resolved_policy_dim(
            self.robot.action_dim
        )
        if policy_action_dim != self.residual_rl.action_dim:
            raise ConfigurationError(
                "action_adapter policy_dim must match residual_rl.action_dim"
            )
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ConfigurationError("camera names must be unique")
        missing_cameras = sorted(
            set(self.robot.action_adapter.required_cameras) - set(names)
        )
        if missing_cameras:
            raise ConfigurationError(
                "action adapter requires undeclared cameras: "
                + ", ".join(missing_cameras)
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the immutable configuration to JSON-compatible data."""

        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "StackConfig":
        """Load a stack configuration from JSON without a YAML dependency."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        robot_data = dict(data.get("robot", {}))
        action_adapter = ActionAdapterConfig(**robot_data.pop("action_adapter", {}))
        config = cls(
            residual_rl=ResidualRLConfig(**data.get("residual_rl", {})),
            speculative=SpeculativeConfig(**data.get("speculative", {})),
            rtc=RTCConfig(**data.get("rtc", {})),
            robot=RobotConfig(action_adapter=action_adapter, **robot_data),
            cameras=tuple(CameraConfig(**value) for value in data.get("cameras", [{}])),
        )
        config.validate()
        return config
