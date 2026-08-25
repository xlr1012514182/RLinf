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

"""Run a bounded CUDA Graph component smoke and emit machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from rlinf.projects.fibocom_vla.inference.cuda_graph import ShapeStableCUDAGraph


def _git_revision() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return revision, dirty


def _summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def _measure_cuda(function: Callable[[Tensor], Tensor], value: Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function(value)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _build_model(width: int, depth: int, dtype: torch.dtype) -> nn.Module:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(width, width), nn.GELU()))
    return nn.Sequential(*layers).to(device="cuda", dtype=dtype).eval()


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the component benchmark using a synthetic shape-stable graph."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this command needs a CUDA device")
    if min(args.batch_size, args.width, args.depth, args.warmup, args.repetitions) <= 0:
        raise ValueError("all benchmark dimensions and counts must be positive")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = _build_model(args.width, args.depth, dtype)
    value = torch.randn(args.batch_size, args.width, device="cuda", dtype=dtype)
    graph = ShapeStableCUDAGraph(
        model,
        warmup_iterations=args.capture_warmup,
        eager_fallback=True,
        clone_outputs=True,
    )
    capture_report = graph.capture(value)
    if not capture_report.captured:
        raise RuntimeError(f"CUDA Graph capture failed: {capture_report.reason}")
    exactness = graph.verify_exactness(value)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(value)
            graph(value)
        torch.cuda.synchronize()
        eager_samples = [_measure_cuda(model, value) for _ in range(args.repetitions)]
        graph_samples = [_measure_cuda(graph, value) for _ in range(args.repetitions)]
    eager_summary = _summary(eager_samples)
    graph_summary = _summary(graph_samples)
    revision, dirty = _git_revision()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": 1,
        "benchmark_scope": "synthetic_shape_stable_component_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "git_dirty": dirty,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": device.name,
            "device_total_memory_bytes": device.total_memory,
        },
        "protocol": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "width": args.width,
            "depth": args.depth,
            "dtype": args.dtype,
            "capture_warmup": args.capture_warmup,
            "measurement_warmup": args.warmup,
            "repetitions": args.repetitions,
            "timing_clock": "torch.cuda.Event",
            "graph_measurement_includes_input_copy_and_output_clone": True,
        },
        "capture": asdict(capture_report),
        "exactness": asdict(exactness),
        "eager": eager_summary,
        "cuda_graph": graph_summary,
        "speedup_from_mean": (eager_summary["mean_ms"] / graph_summary["mean_ms"]),
        "claim_boundary": (
            "This synthetic component smoke is not a pi0.5 latency or robot-quality result."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the finite benchmark command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser


def main() -> int:
    """Run the benchmark and write one self-describing JSON artifact."""

    args = build_parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
