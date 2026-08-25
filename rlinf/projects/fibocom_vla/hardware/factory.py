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

"""Configuration-driven factories with no eager optional imports."""

from __future__ import annotations

from ..config import CameraConfig, RobotConfig
from ..contracts import CameraBackend, RobotBackend
from ..errors import ConfigurationError


def create_robot(config: RobotConfig) -> RobotBackend:
    """Create one robot backend from a validated configuration."""

    config.validate()
    if config.backend == "mock":
        from .mock import MockJointRobot

        return MockJointRobot(config)
    if config.backend == "so101":
        from .so101 import SO101Robot

        return SO101Robot(config)
    if config.backend == "dobot_nova":
        from .dobot_nova import DobotNovaRobot

        return DobotNovaRobot(config)
    if config.backend == "ros2_joint":
        from .ros2_joint import ROS2JointRobot

        return ROS2JointRobot(config)
    raise ConfigurationError(f"unsupported robot backend: {config.backend}")


def create_cameras(
    configs: tuple[CameraConfig, ...]
) -> dict[str, CameraBackend]:
    """Create a name-keyed camera map."""

    cameras: dict[str, CameraBackend] = {}
    for config in configs:
        config.validate()
        if config.name in cameras:
            raise ConfigurationError(f"duplicate camera name: {config.name}")
        if config.backend == "mock":
            from .mock import MockCamera

            camera = MockCamera(config)
        elif config.backend == "opencv":
            from .cameras import OpenCVCamera

            camera = OpenCVCamera(config)
        elif config.backend == "realsense":
            from .cameras import RealSenseCamera

            camera = RealSenseCamera(config)
        elif config.backend == "ros2_image":
            from .cameras import ROS2ImageCamera

            camera = ROS2ImageCamera(config)
        else:
            raise ConfigurationError(f"unsupported camera backend: {config.backend}")
        cameras[config.name] = camera
    return cameras
