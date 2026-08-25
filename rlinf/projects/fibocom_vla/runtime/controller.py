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

"""Closed-loop controller with inference/execution overlap and safe teardown."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from ..config import RTCConfig
from ..contracts import ActionChunk, ChunkPolicy, Observation, RobotBackend, RobotState
from ..hardware.cameras import SynchronizedCameraRig
from ..inference.rtc import AsynchronousChunkPlanner
from ..metrics import RuntimeMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeResult:
    """Bounded run outcome and metrics snapshot."""

    control_steps: int
    chunks: int
    stopped_early: bool
    stop_reason: str
    metrics: dict[str, object]


class ObservationSource:
    """Join robot state, synchronized camera frames, and instruction text."""

    def __init__(
        self,
        robot: RobotBackend,
        cameras: SynchronizedCameraRig,
        instruction: str,
    ) -> None:
        self.robot = robot
        self.cameras = cameras
        self.instruction = instruction
        self._frame_id = 0

    def capture(self, timeout_s: float = 1.0) -> Observation:
        """Capture one observation with explicit state/camera time offsets."""

        state = self.robot.read_state()
        images, camera_timestamp_ns, metadata = self.cameras.read(timeout_s)
        observation = Observation(
            state=state,
            images=images,
            instruction=self.instruction,
            timestamp_ns=camera_timestamp_ns,
            frame_id=self._frame_id,
            metadata={
                **metadata,
                "state_timestamp_ns": state.timestamp_ns,
                "state_camera_offset_ms": (
                    state.timestamp_ns - camera_timestamp_ns
                )
                / 1_000_000.0,
            },
        )
        self._frame_id += 1
        return observation


class RealtimeController:
    """Execute action chunks while planning their successor asynchronously."""

    def __init__(
        self,
        robot: RobotBackend,
        cameras: SynchronizedCameraRig,
        policy: ChunkPolicy,
        rtc_config: RTCConfig,
        *,
        instruction: str,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self.robot = robot
        self.cameras = cameras
        self.policy = policy
        self.rtc_config = rtc_config
        self.metrics = metrics or RuntimeMetrics()
        self.observations = ObservationSource(robot, cameras, instruction)

    @staticmethod
    def _committed_copy(chunk: ActionChunk, committed_prefix: int) -> ActionChunk:
        return ActionChunk(
            values=chunk.values,
            period_s=chunk.period_s,
            source_observation_ns=chunk.source_observation_ns,
            generated_ns=chunk.generated_ns,
            committed_prefix=committed_prefix,
            metadata=chunk.metadata,
        )

    def run(
        self,
        *,
        maximum_control_steps: int,
        stop_condition: Callable[[RobotState], bool] | None = None,
    ) -> EpisodeResult:
        """Run a finite episode and always stop/disconnect on failure."""

        if maximum_control_steps <= 0:
            raise ValueError("maximum_control_steps must be positive")
        steps = 0
        chunks = 0
        stopped_early = False
        stop_reason = "maximum_control_steps"
        self.robot.connect()
        self.cameras.connect()
        planner = AsynchronousChunkPlanner(
            self.policy, self.rtc_config, metrics=self.metrics
        )
        try:
            first_observation = self.observations.capture()
            planner.submit(0, first_observation)
            current = planner.result().output.action
            request_id = 1
            while steps < maximum_control_steps:
                chunk_steps = min(
                    self.rtc_config.execution_horizon,
                    current.horizon,
                    maximum_control_steps - steps,
                )
                trigger_step = max(0, chunk_steps - self.rtc_config.queue_threshold)
                submitted = False
                committed = self._committed_copy(current, chunk_steps)
                next_deadline = time.perf_counter()
                for local_step in range(chunk_steps):
                    loop_start = time.perf_counter()
                    if local_step == trigger_step and steps + local_step < maximum_control_steps:
                        observation = self.observations.capture()
                        planner.submit(
                            request_id,
                            observation,
                            previous=committed,
                            executed_steps=local_step,
                        )
                        request_id += 1
                        submitted = True
                    self.robot.send_joint_target(
                        current.values[local_step], current.period_s
                    )
                    state = self.robot.read_state()
                    steps += 1
                    if stop_condition is not None and stop_condition(state):
                        stopped_early = True
                        stop_reason = "stop_condition"
                        break
                    next_deadline += current.period_s
                    sleep_s = next_deadline - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    self.metrics.record(
                        RuntimeMetrics.CONTROL_LOOP,
                        (time.perf_counter() - loop_start) * 1_000.0,
                    )
                chunks += 1
                if stopped_early or steps >= maximum_control_steps:
                    break
                if not submitted:
                    observation = self.observations.capture()
                    planner.submit(
                        request_id,
                        observation,
                        previous=committed,
                        executed_steps=chunk_steps,
                    )
                    request_id += 1
                current = planner.result().output.action
        except Exception:
            stop_reason = "exception"
            try:
                self.robot.stop()
            except Exception:
                logger.exception("robot stop failed during controller teardown")
            raise
        finally:
            planner.close()
            try:
                self.cameras.disconnect()
            finally:
                self.robot.disconnect()
        snapshot = self.metrics.snapshot()
        return EpisodeResult(
            control_steps=steps,
            chunks=chunks,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            metrics=dict(snapshot),
        )
