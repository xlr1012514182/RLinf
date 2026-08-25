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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError


@dataclass(frozen=True)
class ResidualRLConfig:
    """Frozen-base residual policy and hybrid PPO/twin-Q settings."""

    feature_dim: int = 1024
    action_horizon: int = 50
    action_dim: int = 7
    actor_hidden_dim: int = 866
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
    target_tau: float = 0.005
    second_failure_penalty: float = -1.0
    target_actor_parameters: int = 1_820_000
    parameter_tolerance: int = 10_000

    def validate(self) -> None:
        """Validate shape, probability, and parameter-budget settings."""

        positive = {
            "feature_dim": self.feature_dim,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "actor_hidden_dim": self.actor_hidden_dim,
            "actor_bottleneck_dim": self.actor_bottleneck_dim,
            "max_residual": self.max_residual,
            "parameter_tolerance": self.parameter_tolerance,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if self.action_low >= self.action_high:
            raise ConfigurationError("action_low must be below action_high")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ConfigurationError("gamma and gae_lambda must be in [0, 1]")
        if not 0 < self.ppo_clip_ratio < 1:
            raise ConfigurationError("ppo_clip_ratio must be in (0, 1)")
        if self.ppo_epochs != 1:
            raise ConfigurationError(
                "This reproduction fixes ppo_epochs=1 to match the declared single-round update"
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

        if self.relative_error_threshold < 0 or self.absolute_error_threshold < 0:
            raise ConfigurationError("verification thresholds must be non-negative")
        if self.minimum_prefix < 0:
            raise ConfigurationError("minimum_prefix must be non-negative")
        if not 0 <= self.fallback_acceptance_ratio <= 1:
            raise ConfigurationError("fallback_acceptance_ratio must be in [0, 1]")


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

    def validate(self, action_horizon: int) -> None:
        """Validate overlap and execution horizons."""

        if not 0 < self.execution_horizon <= action_horizon:
            raise ConfigurationError(
                "execution_horizon must be in [1, action_horizon]"
            )
        if not 0 <= self.overlap_horizon <= action_horizon:
            raise ConfigurationError("overlap_horizon exceeds action_horizon")
        if not 0 < self.guidance_decay <= 1:
            raise ConfigurationError("guidance_decay must be in (0, 1]")
        if self.queue_threshold < 0 or self.planner_timeout_s <= 0:
            raise ConfigurationError("RTC queue/timeout settings are invalid")


@dataclass(frozen=True)
class RobotConfig:
    """Backend-neutral robot configuration."""

    backend: str = "mock"
    action_dim: int = 7
    control_hz: float = 20.0
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
    options: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate dimensions and safety envelopes."""

        if self.backend not in {"mock", "so101", "dobot_nova", "ros2_joint"}:
            raise ConfigurationError(f"unsupported robot backend: {self.backend}")
        if self.action_dim <= 0 or self.control_hz <= 0:
            raise ConfigurationError("action_dim and control_hz must be positive")
        vectors = (self.joint_names, self.joint_lower, self.joint_upper, self.max_step)
        if any(len(value) != self.action_dim for value in vectors):
            raise ConfigurationError(
                "joint_names, limits, and max_step must match action_dim"
            )
        if any(low >= high for low, high in zip(self.joint_lower, self.joint_upper)):
            raise ConfigurationError("each joint lower limit must be below upper")
        if any(step <= 0 for step in self.max_step):
            raise ConfigurationError("max_step entries must be positive")


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
    cameras: tuple[CameraConfig, ...] = field(
        default_factory=lambda: (CameraConfig(),)
    )

    def validate(self) -> None:
        """Validate all nested configuration and cross-field invariants."""

        self.residual_rl.validate()
        self.speculative.validate()
        self.rtc.validate(self.residual_rl.action_horizon)
        self.robot.validate()
        for camera in self.cameras:
            camera.validate()
        if self.robot.action_dim != self.residual_rl.action_dim:
            raise ConfigurationError(
                "robot.action_dim must match residual_rl.action_dim"
            )
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ConfigurationError("camera names must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Convert the immutable configuration to JSON-compatible data."""

        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "StackConfig":
        """Load a stack configuration from JSON without a YAML dependency."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(
            residual_rl=ResidualRLConfig(**data.get("residual_rl", {})),
            speculative=SpeculativeConfig(**data.get("speculative", {})),
            rtc=RTCConfig(**data.get("rtc", {})),
            robot=RobotConfig(**data.get("robot", {})),
            cameras=tuple(CameraConfig(**value) for value in data.get("cameras", [{}])),
        )
        config.validate()
        return config
