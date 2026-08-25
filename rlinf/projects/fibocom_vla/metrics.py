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

"""Latency and execution metrics with explicit clock semantics."""

from __future__ import annotations

import math
import statistics
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Mapping


@dataclass(frozen=True)
class LatencySummary:
    """Descriptive latency statistics in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    maximum_ms: float


class RuntimeMetrics:
    """Thread-safe metrics recorder.

    Metric namespaces deliberately separate model compute, robot blocking/idle,
    and end-to-end loop time. RTC may hide model latency behind execution; it
    must never overwrite the underlying model latency measurement.
    """

    MODEL_LATENCY = "model_latency_ms"
    ROBOT_WAIT = "robot_wait_ms"
    CONTROL_LOOP = "control_loop_ms"
    OBSERVATION_AGE = "observation_age_ms"

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, name: str, value: float) -> None:
        """Record one finite scalar sample."""

        if not math.isfinite(value) or value < 0:
            raise ValueError(f"metric {name!r} must be finite and non-negative")
        with self._lock:
            self._values[name].append(float(value))

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named event counter."""

        with self._lock:
            self._counters[name] += amount

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Measure a synchronous code region with ``perf_counter``."""

        start = perf_counter()
        try:
            yield
        finally:
            self.record(name, (perf_counter() - start) * 1_000.0)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if len(values) == 1:
            return values[0]
        position = fraction * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    def summary(self, name: str) -> LatencySummary:
        """Summarize one metric without mutating its samples."""

        with self._lock:
            values = sorted(self._values.get(name, ()))
        if not values:
            return LatencySummary(0, 0.0, 0.0, 0.0, 0.0)
        return LatencySummary(
            count=len(values),
            mean_ms=statistics.fmean(values),
            p50_ms=self._percentile(values, 0.50),
            p95_ms=self._percentile(values, 0.95),
            maximum_ms=values[-1],
        )

    def snapshot(self) -> Mapping[str, object]:
        """Return immutable summaries and counters for logging."""

        with self._lock:
            names = tuple(self._values)
            counters = dict(self._counters)
        return {
            "latencies": {name: self.summary(name) for name in names},
            "counters": counters,
        }
