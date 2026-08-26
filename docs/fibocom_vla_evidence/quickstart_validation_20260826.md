# Fresh-clone Quick Start validation — 2026-08-26

This record covers a bounded CPU-only reproduction of the root README Quick
Start at code commit `627f53006f31906b453658cf91347b98b40cd7ee` on an isolated
Linux x86_64 environment. The outcome is `runnable-smoke`: the
repository-local bootstrap, configuration validation, finite Mock runtime and
Fibocom component tests all ran successfully after the source was downloaded.

This is not an OpenPI checkpoint forward, CUDA/TensorRT benchmark, simulator
task evaluation, ROS2 integration test, or physical-robot validation.

## Source and transport

- Target repository: `https://github.com/xlr1012514182/RLinf.git`
- Target branch: `feat/fibocom-vla-stack`
- Tested commit: `627f53006f31906b453658cf91347b98b40cd7ee`
- Task checkout: isolated under
  `/root/autodl-tmp/codex_fibocom_quickstart_20260826_627f5300_mirror/RLinf`

Direct GitHub transport was not usable from the instance during final
validation. A compatibility mirror was therefore used only as the Git
transport. The recorded checkout identity is the exact target commit above,
and the tested README/script hashes match the inventory below. The verified
claim is that the documented commands work **after the repository has been
downloaded**; this record does not claim that every network can always reach
GitHub directly.

## Bootstrap environment

The README bootstrap created `.venv-fibocom` inside the checkout and installed:

| Item | Observed |
|---|---|
| Python | 3.11.14, repository-local managed runtime |
| `uv` | 0.12.6, repository-local tool |
| PyTorch | 2.6.0+cpu, official CPU wheel |
| NumPy | 1.26.4 |
| RLinf Transformers fork | `rlinf-transformer-openpi==4.53.2` |
| Pinned packages | 32 exact entries |
| CUDA use | None |

The host had an unrelated package-index mirror configured. The bootstrap
overrode it for deterministic tool/package resolution. The default upstream
Python download was also unreachable from this instance, so the script's
documented `FIBOCOM_PYTHON_INSTALL_MIRROR` default supplied the same
Python-build-standalone artifact through the repository's existing GitHub
compatibility mirror precedent.

The recorded final inventory shows the host RTX 4080 SUPER at 0 MiB allocated
and 0% utilization. This CPU-only protocol did not launch a GPU workload.

## Commands and results

The following post-download commands were copied from the root README and run
in order without activating a shell environment:

```bash
bash requirements/fibocom_vla_quickstart.sh

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16

.venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla
```

| Gate | Result |
|---|---|
| Bootstrap | Completed; managed Python and all 32 pinned dependencies installed |
| Config validation | Completed; H=50 / environment action dimension 14 contract accepted |
| Finite Mock runtime | Completed; 16 control steps, 2 chunks, `stop_reason=maximum_control_steps` |
| Fibocom component suite | 322 collected; 317 passed; 5 skipped; 9.32 s |

The Mock log contains synthetic loop timings. They are plumbing observations,
not π0.5, GPU, simulator, or robot performance measurements.

## Evidence inventory

The byte-preserved run artifacts are stored under
`docs/fibocom_vla_evidence/remote_627f5300_quickstart/`. Hashes for the tested
README and bootstrap source files at commit `627f5300` are also recorded in the
inventory below.

| Artifact | SHA-256 |
|---|---|
| `README.md` | `39f73cf4c050b11af51fc947847b965e8f29811b3ee8c8fc2d4c8ae86e143fcd` |
| `README_ZH.md` | `b03dc942bb990f38553efbe2eb30776f5c9c8410c4bb00d86fae5819bed09d5f` |
| `requirements/fibocom_vla_quickstart.sh` | `3f1eea6806a10cf191b9ac577909a3286d6942980ff0fe63de053085dfa611c4` |
| `requirements/fibocom_vla_quickstart.txt` | `0f5671e884d3967b06b9ec51ca9e23f5c0dacd8c7c6b0120347dd7790991a480` |
| `quickstart_bootstrap.log` | `88cb222a84cadb0f79dde6091718aba4258633e064dbaad893fdfdd7584ec75a` |
| `quickstart_validate.log` | `4f57de909baa0db876847166e7fe8e429074942e2010eff5ccbe7c5786354f3a` |
| `quickstart_mock.log` | `affa7d1fa7326c8c4f6b87c450164cea25fe9426570ab5173b5da4659a3832c7` |
| `quickstart_pytest.log` | `0c97fd3f3a2080c8fcf975617fb9af2ddd62e91b4d1a0c65a06db4cfaf4c8a65` |
| `quickstart_environment.txt` | `5f9559eec7981ff3a13dc8059e72c75d86f8157053e731e4fe49ddae0876e84f` |

Still unverified by this protocol: the résumé success-rate, latency and
speedup values; full model/checkpoint downloads; real π0.5 or Draft execution;
CUDA Graph or TensorRT execution; RoboTwin task quality; ROS2 endpoints;
SO101/Nova calibration; and physical motion.
