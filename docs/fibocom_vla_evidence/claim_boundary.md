# Claim-to-evidence boundary

This file prevents implementation capability, authenticated assets, upstream
results, historical experiments, synthetic smokes, and new task measurements
from being merged into one claim. The implementation snapshot reviewed here is
`b75f92b4188d3a7276d772a5e0af09c252fe1790`.

## Current verified floor

- The local Fibocom suite collected **322** cases: **317 passed, 5 skipped**
  (2 warnings).
- The remote Linux/CUDA suite recorded **321 passed, 1 skipped** for the same
  commit; this is component verification, not a model benchmark.
- All **seven** checked-in configuration files parse. Real and retargeting
  templates retain explicit motion/calibration gates.
- The fixed-revision main manifest authenticated all **17** declared files and
  bound `pi05_aloha_robotwin`, `physical-intelligence/robotwin`, H=50, raw
  state/action 14, model state/action 32, and the three RoboTwin camera keys.
- One CUDA forward of the source-locked real main checkpoint passed on one
  **synthetic observation**: 3,616,757,520 parameters; no missing main keys;
  no unresolved unexpected keys; eight complete F32 value-head tensors
  explicitly classified as RLinf training auxiliaries; environment action
  shape `[50, 14]`; model action shape `[50, 32]`; finite outputs; seed 17.
  This is a component smoke, not a task, quality, latency or robot benchmark.

| Résumé statement or capability | Source audit result | Implementation status at `b75f92b4` | Result status |
|---|---|---|---|
| frozen π0.5 + 1.82M residual actor + twin Q | Historical code freezes OpenPI, but its old 1.82M interpretation mixed actor and critic parameters | This branch asserts an approximately 1.82M residual actor and keeps V/twin-Q separate | Parameter geometry and training plumbing are unit-tested; no new policy-quality result |
| binary sparse reward, no shaping, second-failure tail, GAE, one PPO epoch | Historical `stage_binary_guard` contains geometric shaping and the historical PPO path does not close the requested policy-only lineage | Binary task outcomes, explicit second-failure terminal transformation, masked GAE, bounded-policy log-probability and one PPO epoch are implemented | Component-tested only |
| 112 self-rollouts, 75.0%→96.9%, reduced real-robot drops | Audited artifacts contain different episode counts, seeds, guards/routes and IsaacLab/Franka simulation evidence | Archive/collector/checkpoint-lineage interfaces are implemented; no success value is hard-coded | **Not reproduced** as a paired protocol or physical-robot result |
| source-locked π0.5 H=50 RoboTwin checkpoint | The publisher `config.json` declares H=10, while the registered RLinf config and official RLinf YAML declare H=50 | Manifest, stats, transforms, file sizes/SHA-256, state/action geometry and cameras are verified without rewriting publisher bytes | Asset/contract verification and one real-checkpoint H=50 forward passed on one synthetic observation; component-verified only |
| 48.2→42.8 ms CUDA Graph, bit-exact | Historical numbers belong to a StarVLA/Qwen visual path | A reversible OpenPI visual tower/projector graph adapter, stable-signature cache, exactness gate and eager fallback are implemented | No checkpoint-specific CUDA Graph benchmark; **latency not reproduced** |
| TensorRT 1.5×, BF16 fusion rounding | Historical evidence does not close a π0.5 engine result | Checkpoint/profile-bound export, build, immutable engine metadata, runtime-buffer validation and paired/layer audit code are implemented | No engine, parity run, layer ablation or timing artifact; **not reproduced** |
| continuous speculative 58.0→19.1 ms, 3.04×, 94.1→93.8 | Published Draft weights are π0 LIBERO H=50/Aenv=7, and the reported values belong to upstream FLASH protocols | The production factory now requires the concrete verified main target, trained candidate manifest, evaluation report, trusted Ed25519 approval and concrete OpenPI verifier; generic Draft/verifier factories cannot bypass it | No trained/evaluated/signed π0.5 Draft and no paired latency/quality run; **not reproduced** |
| π0 Draft warm-start → π0.5 Draft training | Public π0 weights may initialize compatible query/state/decoder tensors but cannot be promoted directly | A non-signing CLI implements raw-source materialization, exact main-model teacher ownership, hashed cache/run lineage, episode split, real optimizer/resume, validation recomputation and non-production candidate export | Closure is unit-tested with fixtures; no real data cache, optimizer run or candidate exists |
| RTC 5.6 s→4 ms and 76→97 ms | The values use different clock meanings; 5.6 s→4 ms cannot be called model latency without a fixed timing boundary | Native denoising guidance, hard prefix, async scheduling and separate compute/queue/caller-wait/control-loop metrics are implemented | Only an older synthetic VJP smoke exists; no checkpoint/robot timing or quality result |
| SO101/Nova real-robot readiness | The model contract is dual-Aloha state/action14 and cannot be renamed directly to a single-arm 6D/7D command | A strict retargeting boundary, source/camera/timestamp/provenance checks, locked SO101/Nova templates and user-supplied task-engine interface are implemented | Boundary is unit-tested; task-specific FK/IK/gripper conversion, calibration, SDK endpoints and physical validation remain pending |

## Unsupported conclusions

The highest defensible label is **component-verified**. It does not certify:

- any résumé success rate, drop reduction, acceptance rate or task-quality value;
- any stated π0.5, CUDA Graph, speculative, TensorRT or RTC latency/speedup;
- task quality, latency, throughput, stability or robot readiness from the
  single synthetic-observation H=50 checkpoint forward;
- a trained, evaluated or signed production π0.5 Draft;
- a TensorRT engine or BF16 first-divergence localization; or
- calibrated SO101/Nova motion or any physical-robot task result.

Those conclusions require fixed checkpoint/task/seed manifests, raw per-episode
or per-call artifacts, explicit metric clocks, the relevant engine/model
artifacts, and—where applicable—reviewed retargeting/calibration plus a bounded
supervised hardware protocol.
