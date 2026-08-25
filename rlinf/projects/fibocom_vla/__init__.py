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

"""Fibocom VLA post-training and real-time inference stack.

The package is deliberately import-light: CUDA, ROS, LeRobot, and vendor SDKs
are loaded only by the backend that needs them. This keeps configuration and
CPU smoke tests usable on development machines without robot dependencies.
"""

from .contracts import ActionChunk, Observation, PolicyOutput, RobotState

__all__ = ["ActionChunk", "Observation", "PolicyOutput", "RobotState"]
