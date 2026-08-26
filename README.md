# Fibocom π0.5 VLA Post-Training and Inference Runtime Stack

[English](README.md) | [Simplified Chinese](README_ZH.md)

**Frozen-Base Residual RL · Continuous-Action Speculative Inference · CUDA/TensorRT Diagnostics · RTC · Real-Robot Integration**

This branch organizes the source-locked RLinf π0.5 RoboTwin policy into an
auditable post-training and deployment stack.

## What Is Implemented

| Workstream | Implementation | Current Boundary |
|---|---|---|
| **Residual RL Post-Training** | Frozen π0.5 reference policy; a 1,819,897-parameter bounded residual actor at H=50/A=14; independent value head; Twin-Q auxiliary loss; trajectory-level GAE; one-epoch PPO; rollout and checkpoint pipelines | Component-verified |
| **CUDA Graph / TensorRT** | Factory-wired CUDA Graph capture for the vision tower/projector, with a bit-exact gate and eager fallback; TensorRT 10 build/runtime manifests, parity checks, ULP analysis, and layer-by-layer difference diagnostics | Contract-verified |
| **Speculative Inference** | Signed π0.5 Draft gate, single-prefill B×K OpenPI verifier, contiguous-prefix acceptance, same-round main-model takeover, and Torch/Triton post-processing | Production assembly path implemented |
| **RTC** | Asynchronous action-chunk planner, hard freezing of committed prefixes, model-space overlap VJP guidance, exponentially decaying weights, and separate timing measurements | Synthetic components verified |
| **Robot Integration** | SO101, Dobot Nova, ROS2, and camera backends; strict retargeting boundary from dual-Aloha H50×14 actions to SO101-6D/Nova-7D actions | Adapters and locked templates verified |

## Source-Locked Model Target

| Field | Contract |
|---|---|
| Main weights | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed103db0f5549c122f3a17c00ba6426c98` |
| Registered runtime | `pi05_aloha_robotwin`, H=50 |
| Normalization statistics | `physical-intelligence/robotwin`, quantile normalization |
| Geometry | Raw state/action dimension: 14; normalized model state/action dimension: 32 |
| Cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| Draft source | `Dexmal/RealtimeVLA-Flash@77b9a6f8` |

The publisher checkpoint metadata records H=10. This repository does not modify
those bytes. Instead, it verifies the original weights and separately binds the
registered RLinf H=50 runtime, official RoboTwin YAML, normalization statistics,
camera mapping, and Aloha transforms through SHA-256 hashes and source
fingerprints.

## Fresh-Clone Quick Start

The block below is designed for a clean Ubuntu/WSL2 x86_64 checkout and uses
only CPU execution. It requires Git, Bash, `python3` with `pip`, and network
access. It does not require CUDA, model checkpoints, ROS2, or a robot SDK.
Run the complete block from a Bash shell:

```bash
git clone --branch feat/fibocom-vla-stack --single-branch \
  https://github.com/xlr1012514182/RLinf.git
cd RLinf

bash requirements/fibocom_vla_quickstart.sh

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16

.venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla
```

The bootstrap pins Python 3.11.14, CPU PyTorch, the RLinf OpenPI Transformers
fork, and every dependency used by this smoke/test path in a repository-local
`.venv-fibocom`. No shell activation is required. This path verifies
configuration, Mock runtime, and unit-test plumbing only. Full OpenPI/RoboTwin,
CUDA/TensorRT, checkpoint, ROS2, and physical-robot setup requires the
platform-specific prerequisites described in the implementation guide.

## Documentation and Evidence

- [Full English Implementation Guide](examples/embodiment/fibocom_vla/README.md)
- [Full Chinese Implementation Guide](examples/embodiment/fibocom_vla/README_ZH.md)
- [Claim-to-Evidence Boundary](docs/fibocom_vla_evidence/claim_boundary.md)
- [Reproduction Matrix](docs/fibocom_vla_evidence/reproduction_matrix.csv)
- [Pinned Source Identities](docs/fibocom_vla_evidence/source_lock.json)
- [Current Local Validation Record](docs/fibocom_vla_evidence/local_validation_20260826.md)
- [Current Remote Validation Record](docs/fibocom_vla_evidence/remote_validation_20260826.md)
- [Fresh-Clone Quick Start Validation](docs/fibocom_vla_evidence/quickstart_validation_20260826.md)

## Upstream Lineage and License

The implementation base comes from the official
[RLinf repository](https://github.com/RLinf/RLinf). The selected
OpenPI/RoboTwin contracts from the current RLinf codebase are source-locked
independently; see the full implementation guide for details. Code in this
branch follows the parent project's [Apache-2.0 License](LICENSE). Third-party
checkpoints remain subject to their own terms; this repository does not
redistribute or relicense their weights.
