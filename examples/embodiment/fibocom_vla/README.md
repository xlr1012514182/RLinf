# Fibocom π0.5 VLA Stack — Implementation Guide

[简体中文](README_ZH.md) · [Project overview](../../../README.md) · [Testing](../../../docs/fibocom_vla/VALIDATION.md) · [Sources](../../../docs/fibocom_vla/SOURCES.md)

Commands below run from the repository root. CPU examples need no hardware. Model and training examples require the explicitly named external assets; absolute paths under `/opt/fibocom-assets` are examples to replace with your own locations.

## 1. Environment

### CPU components

The Ubuntu/WSL2 x86_64 bootstrap installs Python 3.11.14 and the pinned requirements into `.venv-fibocom`:

```bash
bash requirements/fibocom_vla_quickstart.sh
.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli actor-report \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
.venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla
```

`mock-smoke` selects a Mock policy and requires Mock robot/camera backends. It does not load π0.5. The bootstrap is a Linux script, not a native-Windows full-stack installer.

### OpenPI / GPU

Use a separate Linux environment for the full RLinf/OpenPI stack:

```bash
bash requirements/install.sh embodied --model openpi --env robotwin --install-rlinf
```

The installer binds `rlinf-openpi==0.1.1` and the [OpenPI model requirements](../../../requirements/embodied/models/openpi.txt). Subsequent `python` commands refer to this environment, not automatically to the CPU venv. Install robot SDKs, ROS2 and camera drivers separately when selecting those backends. Do not upgrade individual model dependencies independently of the pinned environment.

## 2. Model contract and assets

| Item | Main policy |
|---|---|
| Checkpoint | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` |
| Revision | `fa8df6ed103db0f5549c122f3a17c00ba6426c98` |
| Registered config | `pi05_aloha_robotwin` |
| Runtime geometry | H=50; raw state/action=14; model state/action=32 |
| Stats | `physical-intelligence/robotwin/norm_stats.json`, quantile `q01/q99` |
| Observation cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| Model image keys | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| Environment output | Absolute Aloha joint/gripper targets after inverse delta and Aloha transforms |

The input/output pipeline uses Aloha joint/gripper conversion, normalized/padded state, and delta transforms for 12 arm joints but not the two grippers. The publisher's H=10 `config.json` remains byte-for-byte intact. The [main manifest](assets/rlinf_pi05_robotwin_adjust_bottle_h50.json) separately fingerprints the registered H=50 config, official RoboTwin YAML, and transform sources. The factory rejects loaded-model or stack geometry mismatches. Do not substitute the unrelated state-15/action-17 `fibocom_mobile` statistics.

With the Hugging Face CLI available in the model environment, download the complete pinned repository (the manifest covers 17 files, including all weight shards):

```bash
MAIN_ROOT=/opt/fibocom-assets/RLinf-Pi05-RoboTwin-SFT-adjust_bottle-fa8df6e
hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  --revision fa8df6ed103db0f5549c122f3a17c00ba6426c98 \
  --local-dir "$MAIN_ROOT"

python -m rlinf.projects.fibocom_vla.cli verify-checkpoint-assets \
  --manifest examples/embodiment/fibocom_vla/assets/rlinf_pi05_robotwin_adjust_bottle_h50.json \
  --root "$MAIN_ROOT" \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
```

Verification checks sizes, SHA-256 hashes, normalization/config dimensions, source fingerprints and stack semantics. It authenticates assets; it does not execute a forward pass. The code license does not grant a license to these weights; see [source terms](../../../docs/fibocom_vla/SOURCES.md).

## 3. Main-policy inference without robot motion

Bind the verified assets to the default factory:

```bash
export FIBOCOM_PI05_MODEL_PATH="$MAIN_ROOT"
export FIBOCOM_PI05_ASSET_MANIFEST="$PWD/examples/embodiment/fibocom_vla/assets/rlinf_pi05_robotwin_adjust_bottle_h50.json"
export FIBOCOM_PI05_CONFIG_NAME=pi05_aloha_robotwin
export FIBOCOM_PI05_DEVICE=cuda
export FIBOCOM_PI05_DENOISE_STEPS=5

python -m rlinf.projects.fibocom_vla.cli openpi-checkpoint-smoke \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json \
  --seed 17

python -m rlinf.projects.fibocom_vla.cli run \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json \
  --instruction "adjust the bottle" --steps 16
```

The first command uses synthetic images/state to check model loading and forward execution. The second combines the real model with Mock robot/cameras. Neither is a RoboTwin task evaluation. In manifest mode, config name can be derived from the manifest; an explicit value must match. Do not set `FIBOCOM_WRIST_CAMERA` to override its ordered camera contract.

## 4. Policy profiles and RTC

| Profile | Configuration | Required artifact |
|---|---|---|
| Base | `residual_rl.enabled=false`, `speculative.enabled=false` | Verified main policy |
| Residual | `residual_rl.enabled=true`, `speculative.enabled=false` | Matching residual checkpoint and checksum sidecar |
| Speculative | `residual_rl.enabled=false`, `speculative.enabled=true` | π0.5 Draft candidate, evaluation, approval and trusted public key |

Keep the configuration stable across training, rollout collection and deployment: checkpoint/config hashes are checked. Residual and speculative profiles cannot be combined.

RTC defaults use an execution horizon of 8, a queue threshold of 2, and an overlap horizon of 16. The planner overlaps generation with execution and freezes committed prefixes. Native OpenPI RTC carries normalized same-forward actions in `ActionChunk.model_values` into model-space overlap guidance. Residual/speculative wrappers and nonlinear retargeting use a host output-space fallback, not native VJP guidance. A timed-out in-flight accelerator call requires restarting its owning process before reusing that model/GPU context.

## 5. Residual post-training

The base remains frozen. A bounded residual actor modifies the base action chunk; an independent V head and Twin-Q auxiliary critic support GAE and one-epoch PPO. The H50×14 profile has 1,819,897 actor parameters (hidden 447, bottleneck 512, feature dimension 2048).

Create your own `artifacts/residual_config.json` from the H50 dry-run template, set `residual_rl.enabled=true`, leave speculative disabled and hardware in Mock/dry-run mode, then validate it. `artifacts/` is ignored by Git. Use that **same config** for initialization, collection, update, and model assembly.

```bash
python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config artifacts/residual_config.json

python -m rlinf.projects.fibocom_vla.rl.train_cli initialize \
  --config artifacts/residual_config.json \
  --output artifacts/residual_init.pt \
  --base-model RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  --base-model-revision fa8df6ed103db0f5549c122f3a17c00ba6426c98 \
  --seed 17 --device cpu

# After collecting a compatible rollout with this behavior checkpoint:
python -m rlinf.projects.fibocom_vla.rl.train_cli update \
  --config artifacts/residual_config.json \
  --input-checkpoint artifacts/residual_init.pt \
  --rollout artifacts/rollout.npz \
  --output artifacts/residual_updated.pt --device cpu
```

These commands operate on residual state and archives, not the robot backend or π0.5 weights. Base-model identity/revision are operator-supplied metadata; bind them to the separately verified main assets. Updates reject mismatched config, behavior checkpoint, base identity, and rollout geometry.

The [rollout recorder](../../../rlinf/projects/fibocom_vla/rl/collector.py) is an integration API, not an automatic task collection runner. Supply task observations, rewards, termination/truncation and behavior-policy data through your collector. Training collection must use stochastic residual sampling and retain its log-probabilities; the deployment factory defaults to deterministic residual inference. To load an updated checkpoint, set `FIBOCOM_RESIDUAL_CHECKPOINT` to its absolute path and retain the generated checksum sidecar. Evaluate the frozen baseline and updated policy under the same task protocol.

## 6. Draft training and deployment

The [published Draft registry](assets/dexmal_flash_pi0_libero_drafts.json) describes **π0 LIBERO**, not a ready-to-use π0.5 RoboTwin Draft:

```bash
DRAFT_ROOT=/opt/fibocom-assets/Dexmal-RealtimeVLA-Flash-77b9a6f
hf download Dexmal/RealtimeVLA-Flash draft_libero_10.pt \
  --revision 77b9a6f88fb100230bc78cb4cb361bd2e586f9fb \
  --local-dir "$DRAFT_ROOT"

python -m rlinf.projects.fibocom_vla.cli verify-draft-asset \
  --registry examples/embodiment/fibocom_vla/assets/dexmal_flash_pi0_libero_drafts.json \
  --root "$DRAFT_ROOT" --suite libero_10
```

Other registry suites are `libero_goal`, `libero_object`, and `libero_spatial`. Verification authenticates the original asset and reports `pi05_direct_use: false`. π0.5 derivation copies compatible warm-start parameters and initializes a new action head.

The [Draft CLI](../../../rlinf/projects/fibocom_vla/draft_train_cli.py) exposes separate stages:

| Command | Inputs / output |
|---|---|
| `materialize` | Task source adapter (`module:function`), source manifest, main-model-owned teacher and split seed → teacher cache |
| `verify-cache` | Cache manifest → content and lineage checks |
| `train` | Cache manifest, `DraftTrainingConfig`, original Draft registry/root/suite → optimizer run |
| `verify-run` | Run manifest → checkpoint and run checks |
| `export-candidate` | Verified cache/run and warm-start identity → unsigned candidate |

```bash
python -m rlinf.projects.fibocom_vla.draft_train_cli materialize --help
python -m rlinf.projects.fibocom_vla.draft_train_cli train --help
python -m rlinf.projects.fibocom_vla.draft_train_cli export-candidate --help
```

Supply a task source adapter and training JSON matching [the training schema](../../../rlinf/projects/fibocom_vla/inference/pi05_draft_training.py). The training code revision must match a clean Git checkout; import a ZIP into a real, reviewed Git commit before training. Candidate export sets `production_authorized=false` and does not sign approvals. A held-out, content-bound evaluation and independent Ed25519 approval are required by the production loader.

For an approved π0.5 candidate, enable the speculative profile and provide:

```bash
export FIBOCOM_PI05_DRAFT_MANIFEST=/absolute/path/candidate/manifest.json
export FIBOCOM_PI05_DRAFT_EVALUATION=/absolute/path/evaluation.json
export FIBOCOM_PI05_DRAFT_APPROVAL=/absolute/path/approval.json
export FIBOCOM_PI05_DRAFT_PUBLIC_KEY=/absolute/path/ed25519_public_key.raw
export FIBOCOM_PI05_DRAFT_SIGNING_KEY_ID=reviewed-key-id
```

The public key is 32 raw bytes. Optional defaults are `FIBOCOM_PI05_DRAFT_DTYPE=bfloat16`, `FIBOCOM_PI05_SPECULATIVE_BACKEND=auto`, and `FIBOCOM_PI05_SPECULATIVE_VERIFICATION_TIMES=0.10,0.05`. The factory constructs the checkpoint-bound parallel verifier itself; no separate Draft/verifier plugin factory is required for this production route. Runtime logic accepts a contiguous prefix and lets the main model take over within the same round when acceptance is insufficient.

## 7. CUDA Graph and TensorRT

Visual capture wraps only the PaliGemma vision tower/projector. With CUDA and an eligible non-RTC, non-speculative, inference-only profile:

```bash
export FIBOCOM_PI05_CUDA_GRAPH_VISUAL=true
export FIBOCOM_PI05_CUDA_GRAPH_CACHE_CAPACITY=4
export FIBOCOM_PI05_CUDA_GRAPH_WARMUP=3
```

Shape eligibility, bit-exactness checks and eager fallback remain active. The checked-in H50 profile enables RTC, so setting this variable alone does not activate capture there. TensorRT utilities accept a supplied ONNX graph or network, build an engine manifest, and report parity/ULP/layer differences; the source release does not include a turnkey whole-policy π0.5 export path or engine. See [validation procedures](../../../docs/fibocom_vla/VALIDATION.md) for the separate synthetic benchmark and model timing protocol.

## 8. Robot integration

| Template | Purpose |
|---|---|
| [mock.json](config/mock.json) | CPU Mock components and control loop |
| [robotwin_pi05_h50_dry_run.json](config/robotwin_pi05_h50_dry_run.json) | H50×14 main-policy contract with Mock robot/cameras |
| [SO101 H50 retargeting](config/robotwin_pi05_h50_so101_6d_retargeting_locked.json) | Locked H50×14 → SO101 six-coordinate interface |
| [Nova H50 retargeting](config/robotwin_pi05_h50_nova_7d_retargeting_locked.json) | Locked H50×14 → Nova seven-coordinate interface |
| [so101_realsense.json](config/so101_realsense.json) | SO101 / RealSense backend template |
| [dobot_nova_ros2.json](config/dobot_nova_ros2.json) | Dobot TCP backend with ROS2 cameras |
| [ros2_nova5_arm.json](config/ros2_nova5_arm.json) | Six-axis ROS2 joint backend template |

SO101 has five arm joints plus a gripper. Nova's seven-coordinate path has six arm joints plus a gripper and requires independent digital-input gripper feedback. Generic affine mapping cannot turn dual-Aloha actions into a calibrated single-arm controller. Implement the task-specific retargeting engine, state encoding, FK/IK, feasibility checks and gripper conversion at the existing adapter boundary.

### Deployment checklist

1. Review robot identities, SDK/serial/TCP endpoints, ROS2 topics/services and camera streams. A hardware `dry_run` may still connect and read devices.
2. Check model camera order, H50×14 state/action semantics, robot joint order, units, scale/offset, limits and control period.
3. Supply genuine calibration, set adapter validation only after review, and match the retargeting engine's calibration ID.
4. Verify limits against hardware before clearing `limits_require_hardware_validation`; do not replace locked placeholders with unverified values.
5. Check camera skew, state/camera skew, observation freshness and action generation age under your deployment load.
6. Check acknowledged stop/emergency-stop behavior, controller interlocks and the physical emergency stop; keep an operator present.
7. Start with bounded dry-run reads. Authorize bounded physical motion only after the above checks, with both `robot.dry_run=false` and `--allow-motion`.

These gates remain part of the runtime. A passing config check is not calibration or motion authorization.

## 9. Tests and source distribution

[Validation procedures](../../../docs/fibocom_vla/VALIDATION.md) cover component tests, model smoke, synthetic profiling and task evaluation. The release carries test/benchmark code but no run-result archive. Keep your own immutable run records outside the source tree. [Sources and licensing](../../../docs/fibocom_vla/SOURCES.md) record the upstream and asset revisions.
