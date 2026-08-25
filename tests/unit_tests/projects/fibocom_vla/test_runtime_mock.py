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

from rlinf.projects.fibocom_vla.config import StackConfig
from rlinf.projects.fibocom_vla.hardware import create_cameras, create_robot
from rlinf.projects.fibocom_vla.hardware.cameras import SynchronizedCameraRig
from rlinf.projects.fibocom_vla.policy import MockChunkPolicy
from rlinf.projects.fibocom_vla.runtime import RealtimeController


def test_mock_runtime_closes_finite_loop() -> None:
    config = StackConfig()
    robot = create_robot(config.robot)
    cameras = SynchronizedCameraRig(create_cameras(config.cameras))
    policy = MockChunkPolicy(
        horizon=config.residual_rl.action_horizon,
        action_dim=config.robot.action_dim,
        period_s=1 / config.robot.control_hz,
    )
    result = RealtimeController(
        robot,
        cameras,
        policy,
        config.rtc,
        instruction="stack the block",
    ).run(maximum_control_steps=5)
    assert result.control_steps == 5
    assert result.chunks == 1
    assert result.stop_reason == "maximum_control_steps"
    assert not robot.is_connected

