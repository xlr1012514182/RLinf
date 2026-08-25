# Remote π0.5 checkpoint validation — 2026-08-26

This record covers a bounded, task-owned Linux/CUDA validation of code commit
`b75f92b4188d3a7276d772a5e0af09c252fe1790` on the user-provided instance. It
closes one real, manifest-bound π0.5 H=50 forward. It is not a latency,
throughput, simulator, success-rate, or physical-robot benchmark.

## Source, assets and isolation

- The tested `git archive` SHA-256 was
  `6db5b4a408cd56fae585602dcf241dbf44f1e501d4b02381cdd1f5e7091da3fe`.
- The task directory and virtual environment were isolated under
  `/root/autodl-tmp/codex_fibocom_h50_20260825`.
- No external process was stopped. GPU compute-process queries were empty
  immediately before the checkpoint run and after it completed.
- The checkpoint remained outside Git and was not modified or redistributed.
- The task-local PaliGemma tokenizer byte hash was
  `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.

The manifest verifier rehashed all 17 declared checkpoint files and returned
`verified=true`. The bound identity was
`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed103db0f5549c122f3a17c00ba6426c98`,
with `pi05_aloha_robotwin`, `physical-intelligence/robotwin`, H=50, raw
state/action 14, model state/action 32 and the three RoboTwin camera keys.

## Environment

| Item | Observed |
|---|---|
| OS | Ubuntu/glibc 2.35, Linux 5.15 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32,760 MiB |
| Driver | 580.105.08 |
| `rlinf-openpi` | 0.1.1 |
| `rlinf-transformer-openpi` | 4.53.2 |
| tokenizers | 0.21.4 |
| JAX / jaxlib | 0.5.3 / 0.5.3 |
| Orbax | 0.11.13 |

The bounded smoke reused the host's task-local Torch 2.8 runtime, while the
`rlinf-openpi` package metadata declares Torch 2.7.1. This is therefore a
cross-version component smoke, not a claim that the environment is the
canonical supported RLinf installation. Unrelated simulator/robot packages
were not installed merely to make package metadata checks green.

## Cross-platform component suite

```text
PYTHONPATH=. python -m pytest -q tests/unit_tests/projects/fibocom_vla
321 passed, 1 skipped in 12.90s
exit code 0
```

## Manifest-bound H=50 forward

Protocol: seed 17, one all-zero synthetic observation, three 224×224 RGB
camera frames, registered `pi05_aloha_robotwin`, one denoising step, BF16 model
parameters, CUDA execution, residual RL disabled and speculative inference
disabled.

Observed result:

| Check | Result |
|---|---|
| CLI status | `smoke_passed=true` |
| Loaded parameter count | 3,616,757,520 |
| Weight source | three verified safetensors shards |
| Main-model missing keys | 0 |
| Main-model unresolved unexpected keys | 0 |
| Checkpoint-only auxiliary | exact eight-key F32 RLinf value head, explicitly classified and ignored |
| Norm stats | `physical-intelligence/robotwin`, quantile mode |
| Environment action | `[50,14]`, finite |
| Normalized model action | `[50,32]`, finite |
| Environment-action SHA-256 | `2609b829d3422a020ca9435175490def3ee413f8e828705bcf278e58a9431ba1` |
| Model-action SHA-256 | `5328492246dc91af1f5c17e642122fb7fa7af2dac5d46b858486ef48b338254e` |
| Policy path | `rlinf_openpi_native` |

The raw load report still exposes all eight `value_head.*` keys. They are not
silently erased from evidence: classification succeeds only for the complete
`1024→512→256→128→1` F32 training head under the exact RoboTwin inference
config. Any partial, extra, wrong-shape, wrong-dtype or forged class remains an
unresolved incompatibility and fails closed.

No timing protocol was run. The output hashes are reproducibility identifiers
for this single synthetic observation, not quality metrics.

## Evidence

Machine-readable artifacts and their SHA-256 inventory are under
`docs/fibocom_vla_evidence/remote_b75f92b4/`. The successful JSON, environment,
unit log, asset-verifier output, pre/post GPU inventories and empty pre/post
compute-process lists are retained separately.

Outcome: `component-verified`.

Still unverified: every résumé success-rate/latency/speedup value, a real
trained/evaluated/signed π0.5 Draft, a checkpoint-specific CUDA Graph benchmark,
a TensorRT engine/parity run, simulator task quality, SO101/Nova calibration,
and physical-robot motion.
