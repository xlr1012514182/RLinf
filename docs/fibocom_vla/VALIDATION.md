# Validation procedures

[中文](VALIDATION_ZH.md) · [Implementation guide](../../examples/embodiment/fibocom_vla/README.md)

This page describes how to run checks and what each checks. It is not a stored run report. Run commands from the repository root and retain outputs in a separate, access-controlled run archive when needed.

## CPU components

After the CPU bootstrap, run:

```bash
for config in examples/embodiment/fibocom_vla/config/*.json; do
  .venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
    --config "$config" || exit 1
done

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli actor-report \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16

.venv-fibocom/bin/python -m pytest -q -rs tests/unit_tests/projects/fibocom_vla
```

Config validation does not connect to the configured robot. `mock-smoke` uses Mock backends. The [unit suite](../../tests/unit_tests/projects/fibocom_vla) covers asset contracts, residual updates, Draft lineage, numerical references, planner lifecycle, freshness, adapters and motion gates. Inspect skipped tests explicitly; a skip is not a passing GPU or hardware check.

## Checks with external prerequisites

| Check | Prerequisites | Interpreting the output |
|---|---|---|
| `verify-checkpoint-assets` | Complete checkpoint download and manifest | File identities, transforms and model/stack contracts; no forward pass |
| `openpi-checkpoint-smoke` | Verified weights and compatible OpenPI/GPU environment | Model loading and one synthetic forward; no task success or latency claim |
| `run` with the H50 Mock template | Same model environment | Model/planner/Mock integration; no physical deployment |
| Synthetic CUDA Graph benchmark | CUDA host and supported PyTorch | Synthetic component timing/parity, not π0.5 end-to-end timing |
| Model acceleration comparison | Actual model inputs, eligible profile, paired eager/optimized runs | Only the explicitly measured timing boundary and parity metric |
| Task evaluation | Task environment/data, exact policy assets, evaluation runner | Only the declared tasks, episodes, seeds and success definition |
| Physical deployment check | Calibrated hardware, reviewed adapters and operator | Only the supervised robot configuration actually exercised |

The implementation guide contains the asset and model commands. Full task evaluation and hardware calibration require a task-specific runner/protocol; the Mock CLI is not a replacement.

## Synthetic CUDA Graph benchmark

In a CUDA environment:

```bash
PYTHONPATH=. python examples/embodiment/fibocom_vla/benchmark_cuda_graph.py \
  --output artifacts/cuda_graph_component.json \
  --batch-size 8 --width 256 --depth 4 --dtype float16 \
  --capture-warmup 3 --warmup 5 --repetitions 50
```

Keep the generated artifact's synthetic designation. It profiles the program's synthetic network, not the downloaded π0.5 checkpoint. For actual-model comparisons, record the input identity, model/engine hash, dtype, device/driver, graph eligibility, warmup/capture treatment, repetitions, synchronization method, and whether the boundary includes preprocessing, transfer, queueing or control. Compare the same inputs and numerical tolerance; report distribution statistics rather than only a fastest sample.

## Task evaluation record

Design the protocol before evaluating:

1. Identify the task/simulator/robot version, split, episode list, seeds and policy checkpoint hashes.
2. Define success, failure, timeout, reset conditions and safety stops without selecting episodes based on favorable outcomes.
3. Keep frozen-base and adapted-policy task conditions comparable; distinguish training/selection data from held-out evaluation data.
4. Retain per-episode outcomes, counts, failed runs and uncertainty, plus relevant video/trajectory references where labels need inspection.
5. For residual learning, bind rollout behavior checkpoint and config identity. For Draft deployment, preserve the evaluated candidate, protocol digest and independent signed approval.

The source release does not supply those results. A code check, synthetic smoke or candidate export should be reported under its own scope.

## Distribution hygiene

Keep runtime manifests, dependency locks, tests, benchmark programs, license notices, and hardware safety gates in the source tree. Keep downloaded weights, generated checkpoints/engines, task data, calibration secrets, logs and run reports outside the public source release. Root `artifacts/`, `runs/`, `checkpoints/`, and test caches are ignored as convenience protections; `.gitignore` does not remove already tracked files and is not a secret scanner.
