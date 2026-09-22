<div align="center">

# Fibocom π0.5 VLA Stack

**Frozen-base adaptation · Continuous-action inference · Robot interfaces**

Built on RLinf

[简体中文](README_ZH.md) · [Implementation guide](examples/embodiment/fibocom_vla/README.md) · [Validation procedures](docs/fibocom_vla/VALIDATION.md) · [Sources and licensing](docs/fibocom_vla/SOURCES.md)

</div>

## Overview

This codebase connects a frozen π0.5 policy to residual post-training, action-chunk inference, and robot control interfaces. Model and deployment contracts are kept separate: weights, normalization, camera order, and action dimensions remain explicit, while policy adaptation and hardware integration are handled by dedicated modules.

The repository distributes **source code, runtime asset manifests, configuration templates, tests, and benchmark programs**. Model weights, task data, deployment calibration, and run artifacts are provided separately.


## Core components

| Module | Interface and scope |
|---|---|
| **Residual learning** | Bounded actor, independent value head and Twin-Q auxiliary critic, GAE, and one-epoch PPO; initialization and update CLIs consume externally collected rollout archives. |
| **Speculative inference** | Continuous-action Draft, single-prefill parallel verification, contiguous-prefix acceptance, and same-round main-model takeover; π0.5 deployment requires a compatible Draft that has been trained, evaluated, and approved with a signature. |
| **RTC** | Asynchronous action-chunk generation, committed-prefix freezing, overlap guidance, and separate checks for observation and generation timestamps. |
| **CUDA Graph** | Graph capture for shape-stable vision towers and projectors, with a bit-exactness gate and eager fallback; RTC, speculative, and gradient-enabled contexts bypass this path. |
| **TensorRT** | Engine manifests, build/runtime utilities, and numerical parity diagnostics; users supply the exported model or network. A complete π0.5 exporter or engine is not included. |
| **Robot interfaces** | SO101, Dobot Nova TCP, and ROS2 joint backends, plus OpenCV, RealSense, and ROS2 cameras; physical deployment requires task-specific mapping and hardware calibration. |

## Quick start: CPU / no hardware

Run from the repository root on **Ubuntu or WSL2 x86_64**. Bash, Git, `python3`, `pip`, and network access are required:

```bash
bash requirements/fibocom_vla_quickstart.sh

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16

.venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla
```

The bootstrap creates `.venv-fibocom` with Python 3.11.14 and pinned CPU dependencies. This path needs no weights, CUDA, robot SDK, or simulator. It checks software contracts and the Mock control loop, not robot task performance. See the [implementation guide](examples/embodiment/fibocom_vla/README.md) for the full OpenPI environment.

## Main policy contract

| Item | Value |
|---|---|
| Main checkpoint | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` |
| Checkpoint revision | `fa8df6ed103db0f5549c122f3a17c00ba6426c98` |
| Registered runtime | `pi05_aloha_robotwin`, action horizon 50 |
| Geometry | Raw state/action 14; padded model state/action 32 |
| Cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| Normalization | `physical-intelligence/robotwin`, quantile `q01/q99` |
| Residual actor | H=50, A=14, hidden=447; 1,819,897 parameters |

The publisher's `config.json` remains unchanged at H=10. The [asset manifest](examples/embodiment/fibocom_vla/assets/rlinf_pi05_robotwin_adjust_bottle_h50.json) separately binds the registered H=50 runtime, source fingerprints, transforms, and file hashes. Downloaded files are not rewritten to reconcile this difference.

## Entry points

- **Understanding and integration:** [English guide](examples/embodiment/fibocom_vla/README.md) / [中文指南](examples/embodiment/fibocom_vla/README_ZH.md).
- **Configuration:** Start with the Mock or H50 dry-run profile in the [safe templates](examples/embodiment/fibocom_vla/config).
- **Residual training:** Use the [training CLI](rlinf/projects/fibocom_vla/rl/train_cli.py) with your own task collector and evaluation protocol.
- **Draft training:** Use the [cache/train/export CLI](rlinf/projects/fibocom_vla/draft_train_cli.py); candidate export and production approval are separate stages.
- **Testing and profiling:** See the [validation procedures](docs/fibocom_vla/VALIDATION.md), [component tests](tests/unit_tests/projects/fibocom_vla), and [synthetic CUDA Graph benchmark](examples/embodiment/fibocom_vla/benchmark_cuda_graph.py).

## Physical deployment

The checked-in real-robot templates remain in dry-run or motion-locked mode. The dual-Aloha H50×14 policy cannot be used directly as an SO101-6D or Nova-7D controller: the retargeting interface requires a task-specific engine, matching calibration, joint units/order/limits, camera timing, valid feedback, and verified stop behavior. Physical motion additionally requires both `dry_run=false` and explicit `--allow-motion`. Follow the guide's checklist before changing either setting.

## Repository layout

```text
rlinf/projects/fibocom_vla/
  rl/          residual policy, rollout archives, GAE/PPO, checkpoints
  inference/   Draft/verifier, RTC, CUDA Graph, TensorRT diagnostics
  runtime/     synchronized control loop and asynchronous planner
  hardware/    robot/camera backends, adapters, retargeting
  assets.py    checkpoint manifests and transform verification
  factories.py model and runtime assembly
  cli.py       config checks, smoke tests, and bounded run entry points
examples/embodiment/fibocom_vla/   guides, configs, asset contracts
tests/unit_tests/projects/fibocom_vla/   component and safety tests
docs/fibocom_vla/                 sources and validation procedures
```

## Sources and licensing

This project is an independent adaptation of [RLinf](https://github.com/RLinf/RLinf), not an official RLinf or Fibocom release. The [source inventory](docs/fibocom_vla/SOURCES.md) retains the implementation base, related designs, model revisions, and SDK references without presenting their experimental results as this project's results.

Repository code uses [Apache-2.0](LICENSE). External SDKs, checkpoints, and datasets are subject to their own licenses. The pinned main-checkpoint manifest records that its model repository did not declare a license; verify the applicable permissions before using or redistributing the weights.
