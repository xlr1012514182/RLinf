# Local component validation — 2026-08-26

This record validates the Fibocom implementation at code commit
`b75f92b4188d3a7276d772a5e0af09c252fe1790`. Only README and evidence files
were uncommitted during the final executable checks; the tested Python source
and tests matched that commit.

This is component and interface evidence. It is not a simulator result, a
physical-robot result, or a reproduction of any résumé success rate or latency.

## Environment

| Item | Observed |
|---|---|
| OS | Windows 10.0.26200, 64-bit |
| Python | 3.10.4 |
| PyTorch | 2.6.0+cpu |
| pytest | 8.3.5 |
| GPU used | no |

## Final focused suite

```text
python -m pytest -q tests/unit_tests/projects/fibocom_vla
322 collected
317 passed, 5 skipped, 2 warnings in 14.99s
exit code 0
```

The skipped cases are environment-dependent CUDA/Triton paths. The suite
covers deterministic contracts, fakes and fail-closed behavior; it cannot
establish policy quality or hardware safety.

The checkpoint-only value-head change was additionally exercised by 44 focused
factory/checkpoint/speculative tests. Complete, partial, extra, wrong-shape,
wrong-dtype and forged auxiliary classifications are covered; only the exact
RoboTwin F32 `1024→512→256→128→1` training head is classifiable.

## Static and configuration gates

```text
python -m ruff check rlinf/projects/fibocom_vla \
  rlinf/models/embodiment/openpi/__init__.py \
  tests/unit_tests/projects/fibocom_vla
All checks passed

python -m ruff format --check <same paths>
86 files already formatted

python -m compileall -q rlinf/projects/fibocom_vla \
  rlinf/models/embodiment/openpi/__init__.py
exit code 0
```

All seven JSON configurations passed `validate-config`:

- `dobot_nova_ros2.json`
- `mock.json`
- `robotwin_pi05_h50_dry_run.json`
- `robotwin_pi05_h50_nova_7d_retargeting_locked.json`
- `robotwin_pi05_h50_so101_6d_retargeting_locked.json`
- `ros2_nova5_arm.json`
- `so101_realsense.json`

The SO101/Nova retargeting templates intentionally remain motion-locked; valid
configuration syntax is not kinematic calibration or physical validation.

## Asset and finite-loop gates

The main manifest verifier rehashed all 17 declared files under the local
fixed-revision checkpoint. It confirmed:

- repository revision `fa8df6ed103db0f5549c122f3a17c00ba6426c98`;
- registered config `pi05_aloha_robotwin` and asset ID
  `physical-intelligence/robotwin`;
- H=50, raw state/action 14, model state/action 32;
- `cam_high`, `cam_left_wrist`, `cam_right_wrist`;
- quantile normalization and the selected norm-statistics file.

All four fixed-revision Dexmal Draft files also passed size/SHA-256 checks.
Their contract remains π0 LIBERO H=50/Aenv=7 and
`pi05_direct_use=false`; they are only eligible as verified warm starts before
π0.5 retraining, evaluation and approval.

The finite Mock loop completed 16 control steps and two chunks with
`stop_reason=maximum_control_steps`. Its timings characterize only the Mock
loop and are not model or robot latency. Residual actor reports remained within
the declared 1.82M tolerance: `so101_realsense.json` (policy/residual action
dimension 7) reported 1,818,542 parameters, while `ros2_nova5_arm.json`
(policy/residual action dimension 6) reported 1,819,139; V/twin-Q modules are
separate.

## Result boundary

The real checkpoint forward is recorded separately in
`remote_validation_20260826.md`. Neither record verifies physical motion,
success rate, CUDA Graph latency, TensorRT speedup, speculative quality/latency,
RTC end-to-end timing, or a trained and signed π0.5 Draft.
