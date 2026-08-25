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

"""Backend-neutral policy adapters and deterministic smoke-test policies."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .contracts import ActionChunk, Observation, PolicyOutput
from .errors import ShapeMismatchError


class CallableChunkPolicy:
    """Adapt a pure callable returning ``[horizon, action_dim]`` values."""

    def __init__(
        self,
        function: Callable[[Observation], np.ndarray],
        *,
        period_s: float,
        path: str = "callable",
    ) -> None:
        if not np.isfinite(period_s) or period_s <= 0:
            raise ValueError("period_s must be finite and positive")
        self.function = function
        self.period_s = period_s
        self.path = path

    def predict(self, observation: Observation) -> PolicyOutput:
        """Time the callable and wrap its action values."""

        start_ns = time.perf_counter_ns()
        values = np.asarray(self.function(observation), dtype=np.float32)
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return PolicyOutput(
            action=ActionChunk(
                values=values,
                period_s=self.period_s,
                source_observation_ns=observation.timestamp_ns,
            ),
            model_latency_ms=latency_ms,
            path=self.path,
        )


class MockChunkPolicy:
    """Deterministic local-target generator for the full runtime smoke path."""

    def __init__(
        self,
        *,
        horizon: int,
        action_dim: int,
        period_s: float,
        amplitude: float = 0.02,
    ) -> None:
        if (
            min(horizon, action_dim, period_s) <= 0
            or not np.isfinite(period_s)
            or not np.isfinite(amplitude)
            or amplitude < 0
        ):
            raise ValueError("invalid mock policy geometry")
        self.horizon = horizon
        self.action_dim = action_dim
        self.period_s = period_s
        self.amplitude = amplitude
        self._call_index = 0

    def predict(self, observation: Observation) -> PolicyOutput:
        """Generate a smooth chunk around the latest robot position."""

        start_ns = time.perf_counter_ns()
        state = observation.state.joint_positions
        if state.shape != (self.action_dim,):
            raise ShapeMismatchError(
                f"mock policy expected {self.action_dim} joints, got {state.shape}"
            )
        phase = np.linspace(0.0, np.pi, self.horizon, dtype=np.float32)
        phase += np.float32(self._call_index * 0.1)
        delta = np.sin(phase)[:, None] * self.amplitude
        values = np.repeat(state[None, :], self.horizon, axis=0) + delta
        self._call_index += 1
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return PolicyOutput(
            action=ActionChunk(
                values=values.astype(np.float32),
                period_s=self.period_s,
                source_observation_ns=observation.timestamp_ns,
            ),
            model_latency_ms=latency_ms,
            path="mock_policy",
            diagnostics={"call_index": self._call_index - 1},
        )
