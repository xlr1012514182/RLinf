# Remote component validation — 2026-08-25

This record covers a bounded smoke run on the user-provided SeetaCloud
instance. It validates commit-bound CPU/Mock contracts plus two synthetic CUDA
components. It is not a π0.5 checkpoint, TensorRT, speculative-policy,
simulator, or physical-robot benchmark.

## Source transfer and isolation

- Source commit: `fb0d04a84d6de49c2e2a2319b0e9bfeb8e43dd6c`
- Local `git archive` SHA-256:
  `37055465fe04f26d220ffec2db57e1fea6e912ef7f222c1d41090d51fbdd37e1`
- Remote task directory:
  `/root/autodl-tmp/codex_fibocom_vla_20260825`
- The remote SHA-256 matched before extraction.
- Two direct GitHub clone attempts failed before checkout (HTTP/2 framing
  error, then port-443 timeout). The test therefore used the verified archive;
  the synthetic CUDA JSON reports `git_revision=unavailable`/`git_dirty=true`
  because a `git archive` intentionally contains no `.git` directory.
- A task-local `--system-site-packages` venv was used. Only pytest 8.3.5 and
  OmegaConf 2.3.0 plus their small dependencies were installed into that venv.
  No global environment, external process, model, dataset, or robot resource
  was modified.

## Environment and resource gates

| Item | Observed |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 4080 SUPER, capability 8.9 |
| Driver | 580.105.08 |
| Reported GPU memory | 32,760 MiB |
| GPU compute processes before/after | 0 / 0 |
| External processes stopped | 0 |

## Executed results

### Cross-platform unit and Mock gates

```text
PYTHONPATH=. python -m pytest -q tests/unit_tests/projects/fibocom_vla
131 passed in 5.20s

PYTHONPATH=. python -m compileall -q rlinf/projects/fibocom_vla
exit code 0

PYTHONPATH=. python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
control_steps=16, chunks=2, stop_reason=maximum_control_steps
```

### Synthetic shape-stable CUDA Graph

Protocol: seed 17, FP16, batch 8, width 256, four MLP layers, three capture
warm-ups, five measurement warm-ups, and 50 CUDA-event repetitions. The graph
captured successfully and eager/graph output was bit-exact (`max_abs=0`,
`max_rel=0`). The synthetic means were 0.150132 ms eager and 0.050680 ms graph
(`2.9623×`). These numbers characterize only this tiny synthetic component;
they must not be quoted as π0.5 or résumé latency.

### Synthetic RTC VJP on CUDA

Protocol: one 50×7 chunk, hard prefix 8, overlap 16, five Euler/VJP steps,
float32. The hard prefix remained bit-exact after every update. All five VJP
steps ran, guidance loss decreased monotonically from 0.376497 to 0.364270,
and guided overlap MSE was 0.793274 versus 0.819674 without guidance. This
validates the generic CUDA/autograd mechanism only; the velocity function was
synthetic and no OpenPI checkpoint was loaded.

## Remaining blocked prerequisites

- no π0.5 checkpoint or identity-bound H=50 OpenPI config;
- no checkpoint-bound draft and normalized model-space parallel verifier;
- no OpenPI visual-subgraph capture integration;
- no π0.5 TensorRT export/build/engine;
- no simulator, SO101, Dobot, ROS2 graph, camera, or physical task evaluation.

Outcome: `component-verified`.

Machine-readable artifacts and their hashes are under
`docs/fibocom_vla_evidence/remote_fb0d04a8/`.
