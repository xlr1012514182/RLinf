# Local validation record — 2026-08-25

This record covers the checked-out `feat/fibocom-vla-stack` worktree based on
RLinf `release/v0.2` (`46213e88caa910a4a52e68bde4fb96416c1efa55`). It is a
component/Mock validation record, not a GPU, checkpoint, simulator, or physical
robot benchmark.

## Environment

- Host OS: Windows
- Python: 3.10.4
- Pytest: 8.3.5
- Validation environment: `work/fibocom_vla_stack/.venv`
- Validation scope: CPU unit tests, static checks, JSON/config contracts, and
  deterministic Mock robot/camera plumbing

The environment emitted one non-fatal PyTorch warning: the installed `optree`
is older than 0.13, so PyTorch's C++ pytree/Dynamo integration is disabled. The
tested Fibocom code paths do not use that optimization.

## Executed gates

### Unit tests

```text
python -m pytest -q tests/unit_tests/projects/fibocom_vla
collected 131 items
131 passed, 1 warning in 7.08s
```

The suite covers residual actor/checkpoint/training provenance, sparse reward
and GAE, archive/collector lineage, CUDA Graph and TensorRT diagnostics,
continuous speculative verification, host/native RTC, OpenPI same-forward
features/model-action lineage, action and kinematics adapters, hardware
freshness/lifecycle gates, factory fail-closed behavior, and the Mock runtime.
It also regresses the stock π0.5 10-step versus stack 50-step horizon mismatch,
first-command acknowledgement failure emergency teardown, and required/allowed
action-adapter schema keys for every checked-in example.

### Style and syntax

```text
ruff check rlinf/projects/fibocom_vla tests/unit_tests/projects/fibocom_vla
All checks passed!

ruff format --check rlinf/projects/fibocom_vla tests/unit_tests/projects/fibocom_vla
61 files already formatted

python -m compileall -q rlinf/projects/fibocom_vla
exit code 0
```

### Runtime configuration contracts

All checked-in JSON files parsed through `StackConfig.from_json`:

| Config | Robot backend | Robot action dim | Policy/residual action dim |
|---|---:|---:|---:|
| `dobot_nova_ros2.json` | `dobot_nova` | 7 | 7 |
| `mock.json` | `mock` | 7 | 7 |
| `ros2_nova5_arm.json` | `ros2_joint` | 6 | 6 |
| `so101_realsense.json` | `so101` | 6 | 7 |

The SO101 6↔7 dimensional difference is intentional and is accepted only
through the declared custom LIBERO kinematics boundary. The checked-in real
robot templates retain placeholder calibration/SDK values and therefore fail
closed before motion.

### Residual actor parameter contracts

```text
so101_realsense.json: 1,818,542 trainable actor parameters
ros2_nova5_arm.json: 1,819,139 trainable actor parameters
target: 1,820,000; configured tolerance: 10,000
```

Twin-Q and the independent state-value critic are separate from those actor
counts.

### Mock control loop

```text
python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
control_steps=16, chunks=2, stopped_early=false,
stop_reason=maximum_control_steps
```

The observed Mock timing values are scheduler/plumbing diagnostics only. They
must not be cited as π0.5, CUDA, TensorRT, speculative, RTC, simulator, or
physical-robot performance.

## Unverified gates

- No π0.5 checkpoint was loaded.
- No draft checkpoint or real parallel verifier was loaded.
- No CUDA Graph, Triton, TensorRT, or GPU latency benchmark was run locally.
- The repository does not yet contain a checkpoint-bound π0.5 draft/verifier,
  an OpenPI visual-subgraph capture wrapper, a production Triton kernel, or a
  π0.5 TensorRT export/build pipeline; their generic contracts and diagnostics
  are component-tested only.
- No SO101, Dobot Nova, ROS2 graph, RealSense camera, simulator, or physical
  task was connected.
- No physical success-rate or résumé performance number was reproduced.

The maximum supported outcome for this record is `component-verified`.
