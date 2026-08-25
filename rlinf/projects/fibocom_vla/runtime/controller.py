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
import sys
import time
from dataclasses import dataclass
from typing import Callable

from ..config import RTCConfig
from ..contracts import ActionChunk, ChunkPolicy, Observation, RobotBackend, RobotState
from ..errors import SafetyViolationError
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
        robot_config = getattr(robot, "config", None)
        if robot_config is None:
            raise TypeError("robot backend must expose its validated config")
        self._maximum_observation_age_ns = int(
            float(robot_config.maximum_observation_age_s) * 1_000_000_000
        )
        self._maximum_state_camera_skew_ns = int(
            float(robot_config.maximum_state_camera_skew_ms) * 1_000_000
        )

    def capture(self, timeout_s: float = 1.0) -> Observation:
        """Capture one observation with explicit state/camera time offsets."""

        state = self.robot.read_state()
        images, camera_timestamp_ns, metadata = self.cameras.read(timeout_s)
        camera_timestamps = tuple(
            int(value) for value in metadata.get("camera_timestamps_ns", {}).values()
        ) or (int(camera_timestamp_ns),)
        timestamps = (int(state.timestamp_ns), *camera_timestamps)
        now_ns = time.monotonic_ns()
        if any(timestamp <= 0 or timestamp > now_ns for timestamp in timestamps):
            raise SafetyViolationError(
                "observation contains an invalid/future monotonic timestamp"
            )
        oldest_age_ns = now_ns - min(timestamps)
        if oldest_age_ns > self._maximum_observation_age_ns:
            raise SafetyViolationError(
                f"observation input is stale: {oldest_age_ns / 1_000_000:.3f} ms"
            )
        state_camera_skew_ns = max(
            abs(int(state.timestamp_ns) - timestamp) for timestamp in camera_timestamps
        )
        if state_camera_skew_ns > self._maximum_state_camera_skew_ns:
            raise SafetyViolationError(
                "robot state/camera skew exceeds gate: "
                f"{state_camera_skew_ns / 1_000_000:.3f} ms"
            )
        observation = Observation(
            state=state,
            images=images,
            instruction=self.instruction,
            timestamp_ns=camera_timestamp_ns,
            frame_id=self._frame_id,
            metadata={
                **metadata,
                "state_timestamp_ns": state.timestamp_ns,
                "state_camera_offset_ms": (state.timestamp_ns - camera_timestamp_ns)
                / 1_000_000.0,
                "maximum_state_camera_skew_ms": state_camera_skew_ns / 1_000_000.0,
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
        robot_config = getattr(robot, "config", None)
        if robot_config is None:
            raise TypeError("robot backend must expose its validated config")
        self._maximum_source_age_ns = int(
            float(robot_config.maximum_source_age_s) * 1_000_000_000
        )
        self._maximum_generation_age_ns = int(
            float(robot_config.maximum_generation_age_s) * 1_000_000_000
        )

    def _validate_action_freshness(self, chunk: ActionChunk) -> None:
        now_ns = time.monotonic_ns()
        if chunk.source_observation_ns > now_ns or chunk.generated_ns > now_ns:
            raise SafetyViolationError(
                "action provenance timestamp is in the future clock domain"
            )
        source_age_ns = now_ns - chunk.source_observation_ns
        generation_age_ns = now_ns - chunk.generated_ns
        if source_age_ns > self._maximum_source_age_ns:
            raise SafetyViolationError(
                "action source observation is stale: "
                f"{source_age_ns / 1_000_000:.3f} ms"
            )
        if generation_age_ns > self._maximum_generation_age_ns:
            raise SafetyViolationError(
                "generated action chunk is stale: "
                f"{generation_age_ns / 1_000_000:.3f} ms"
            )

    @staticmethod
    def _committed_copy(chunk: ActionChunk, committed_prefix: int) -> ActionChunk:
        return ActionChunk(
            values=chunk.values,
            period_s=chunk.period_s,
            source_observation_ns=chunk.source_observation_ns,
            generated_ns=chunk.generated_ns,
            model_values=chunk.model_values,
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
        robot_connected = False
        cameras_connected = False
        send_attempted = False
        planner: AsynchronousChunkPlanner | None = None
        try:
            self.robot.connect()
            robot_connected = True
            self.cameras.connect()
            cameras_connected = True
            first_observation = self.observations.capture()
            if self.rtc_config.enabled:
                planner = AsynchronousChunkPlanner(
                    self.policy, self.rtc_config, metrics=self.metrics
                )
                planner.submit(0, first_observation)
                current = planner.result().output.action
            else:
                first_output = self.policy.predict(first_observation)
                self.metrics.record(
                    RuntimeMetrics.MODEL_LATENCY, first_output.model_latency_ms
                )
                current = first_output.action
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
                    if (
                        self.rtc_config.enabled
                        and local_step == trigger_step
                        and steps < maximum_control_steps
                    ):
                        assert planner is not None
                        observation = self.observations.capture()
                        planner.submit(
                            request_id,
                            observation,
                            previous=committed,
                            executed_steps=local_step,
                        )
                        request_id += 1
                        submitted = True
                    self._validate_action_freshness(current)
                    # A backend may place the command on its transport before
                    # raising while parsing an acknowledgement. Track entry to
                    # the send boundary independently from successful steps so
                    # even a first-send exception selects emergency teardown.
                    send_attempted = True
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
                if not self.rtc_config.enabled:
                    observation = self.observations.capture()
                    output = self.policy.predict(observation)
                    self.metrics.record(
                        RuntimeMetrics.MODEL_LATENCY, output.model_latency_ms
                    )
                    current = output.action
                    continue
                assert planner is not None
                if not submitted:
                    observation = self.observations.capture()
                    planner.submit(
                        request_id,
                        observation,
                        previous=committed,
                        executed_steps=chunk_steps,
                    )
                    request_id += 1
                planned_action = planner.result().output.action
                if planned_action.committed_prefix:
                    if planned_action.committed_prefix >= planned_action.horizon:
                        raise RuntimeError(
                            "RTC result contains no uncommitted action suffix"
                        )
                    planned_action = planned_action.suffix(
                        planned_action.committed_prefix
                    )
                current = planned_action
        except Exception:
            stop_reason = "exception"
            raise
        finally:
            active_exception = sys.exc_info()[0] is not None
            cleanup_errors: list[Exception] = []
            if robot_connected:
                try:
                    if stop_reason == "exception" and send_attempted:
                        self.robot.emergency_stop()
                    else:
                        self.robot.stop()
                except Exception as error:
                    cleanup_errors.append(error)
                    logger.exception("robot halt failed during controller teardown")
            if planner is not None:
                try:
                    planner.close()
                except Exception as error:
                    cleanup_errors.append(error)
                    logger.exception("planner close failed during controller teardown")
            if cameras_connected:
                try:
                    self.cameras.disconnect()
                except Exception as error:
                    cleanup_errors.append(error)
                    logger.exception(
                        "camera disconnect failed during controller teardown"
                    )
            if robot_connected:
                try:
                    self.robot.disconnect()
                except Exception as error:
                    cleanup_errors.append(error)
                    logger.exception(
                        "robot disconnect failed during controller teardown"
                    )
            if cleanup_errors and not active_exception:
                raise RuntimeError("controller teardown failed") from cleanup_errors[0]
        snapshot = self.metrics.snapshot()
        return EpisodeResult(
            control_steps=steps,
            chunks=chunks,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            metrics=dict(snapshot),
        )
