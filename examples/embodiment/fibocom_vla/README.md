# Fibocom π0.5 post-training and real-time inference stack

This project extends RLinf `release/v0.2` with a bounded implementation of the
Fibocom internship work: a frozen π0.5 residual post-training path, CUDA Graph
runtime capture, continuous-action speculative inference, asynchronous RTC,
and pluggable SO101/Dobot Nova/ROS2/camera backends.

The implementation and the benchmark evidence are intentionally separate.
Numbers in a résumé, paper, upstream README, or historical StarVLA experiment
are never emitted as results of a new run. A benchmark becomes a local result
only after its JSON/CSV artifact records the commit, config, device, warm-up,
sample count, metric definition, and quality gate.

## Source alignment

| Source | Audited revision | What is reused |
|---|---|---|
| RLinf | `release/v0.2` / `46213e88` | OpenPI π0.5 loading, transforms, PPO/GAE conventions, real-world abstractions |
| RLinf latest | `230ea79b` | compatibility review only; changes are not silently backported |
| FlashRT | `f72192b2` | CUDA Graph/exactness and acceleration-path design evidence |
| realtime-vla-flash | `da6cecca` | continuous draft, batched verification, prefix acceptance, RTC comparison |
| Last-R1 local archive | SHA-256 recorded under `docs/fibocom_vla_evidence/` | residual RL lineage and conflict audit |
| historical Codex task `019fe3f8-…` | all 95 turns plus 52 hashed artifacts audited | StarVLA profiling/bit-exact lessons, not π0.5 results |

## What the code implements

### Frozen-base residual post-training

- The base π0.5 module is forced to `eval()`, every base parameter has
  `requires_grad=False`, and a pre-update assertion rejects accidental unfreeze.
- The trainable actor receives a 2048-D frozen-prefix feature and the full reference
  action chunk. Its adjusted action is `a_ref + bounded_delta`; feasible
  asymmetric residual bounds keep the final action inside the configured action
  range even when the reference lies near a boundary.
- The 50×7 actor with hidden width 562 has exactly **1,818,542** trainable
  scalars. The 50×6 actor with hidden width 581 has exactly **1,819,139**.
  Both are the audited 1.82M-class geometries; construction fails if the actual
  count falls outside the configured tolerance.
- The actor update is explicitly named `HybridResidualPPOTrainer`: clipped PPO
  with trajectory-level GAE plus a separately weighted twin-Q auxiliary term.
  The code does not pretend that PPO and off-policy double-Q are identical.
- GAE uses an independent state-value head conditioned on frozen-base features
  and the reference chunk; it never substitutes `min Q(s, mode(policy))` for
  `V(s)`. Twin-Q remains an explicitly auxiliary TD critic.
- Every rollout chunk records `action_source` and `policy_mask`. Only residuals
  actually sampled from the bounded behavior distribution carry finite stored
  log-probabilities and enter PPO. Reference, guard, clamp, rescue,
  deterministic, and padding actions are excluded by contract, while valid
  intervention actions may still train the critics.
- Task rewards enter as chunk-level `{0,1}` outcomes. At the second discrete
  failure, that transition becomes terminal with a fixed failure outcome; later
  tail entries are zeroed and masked. The negative GAE propagation is tested.
- `ppo_epochs` is fixed to one for the declared single-round update.

### Runtime acceleration

- `ShapeStableCUDAGraph` captures only a declared stable tensor tree. Shape,
  stride, dtype, device, structure, or non-tensor changes select eager fallback.
  `verify_exactness` requires `torch.equal` and also reports absolute/relative
  errors.
- The continuous speculative path performs draft chunk generation, batched
  main-model verification along an interpolation path, contiguous prefix
  acceptance, and suffix completion. Low acceptance can use the résumé-specific
  same-round full-main takeover; diagnostics label this as an override because
  the audited upstream repository schedules full inference on the next round.
- RTC records model compute, planner queue wait, robot blocking wait, and control
  loop time under different metric names. Hiding compute behind execution does
  not rewrite model latency.
- Native OpenPI RTC keeps the normalized action from the same forward in
  `ActionChunk.model_values`. The next request derives its model-space hard
  prefix and overlap target from that exact lineage, freezes the prefix during
  every denoising step, and applies a differentiable/VJP overlap objective with
  exponentially decayed position weights. Missing or inconsistent model-space
  lineage fails closed.
- Native RTC is capability-gated end to end. The current residual wrapper,
  speculative wrapper, and custom nonlinear LIBERO IK/FK adapter do not expose
  a lineage-safe native RTC path, so the planner uses an explicitly labelled
  host output-space fallback (`rtc_postprocess_fallback=true`,
  `rtc_guidance_mode=host_output_postprocess`). That fallback is plumbing, not
  denoising-time VJP guidance.
- Python cannot safely cancel an in-flight model/GPU forward. If the planner
  worker is still alive after `planner_shutdown_timeout_s`, teardown raises;
  restart the owning process before reusing its model or accelerator context.

### Hardware and sensor backends

- SO101 uses the official LeRobot `SO101Follower` API and its `*.pos` contract.
  It is a five-DOF arm plus gripper (six commanded motors), not a six-DOF arm.
  Consequently, a six-coordinate Cartesian pose delta cannot be relabelled as
  six SO101 joints; the LIBERO example requires a reviewed, task-specific
  kinematics engine that rejects infeasible or ambiguous targets.
- Dobot Nova/NovaLite uses the official TCP/IP V4 SDK, 30004 feedback, and
  six arm targets in degrees through `ServoJ`. The optional seventh coordinate
  is a binary gripper command sent through DO; real seven-dimensional control
  requires independent DI feedback rather than commanded-state echo. Before
  every command, controller errors/collision, the documented active-low
  `SafetyState` interlocks, and all four safety-skin approach fields are checked.
- A generic ROS2 backend uses standard `sensor_msgs/JointState` and
  `trajectory_msgs/JointTrajectory`. Its acknowledged stop path supports either
  `std_srvs/srv/Trigger` (`service_backend=trigger`) or Dobot
  `dobot_msgs_v4/srv/{Stop,EmergencyStop}`
  (`service_backend=dobot_v4`). Real mode preflights the command subscriber and
  both services; it never synthesizes a hold from a possibly stale joint state.
  A six-axis Nova5 arm template is provided at
  `config/ros2_nova5_arm.json`.
- Cameras include OpenCV, RealSense SDK, ROS2 `sensor_msgs/Image`, and Mock.
  Every multi-camera read shares one global timeout budget and must satisfy the
  camera-to-camera skew gate. The joined observation then passes monotonic-time,
  maximum-age, and robot-state/camera-skew gates.
- Every real backend starts in `dry_run`. Before every send, the controller
  checks both source-observation age and action-generation age. Joint
  range/delta checks, non-finite latching, backend feedback freshness, bounded
  episodes, connect rollback, stop/emergency-stop selection, reverse-order
  camera/robot cleanup, and an explicit second CLI motion switch are included.
  Dry-run is read-only: target, stop, and emergency writes are suppressed;
  SO101 skips the high-level configure/calibrate lifecycle, and Dobot rejects
  connection-time request-control, clear-error, or enable writes.

### Policy/robot semantic adapter and calibration gate

`PolicyRobotAdapterPolicy` is installed by both CLI execution paths. It adapts
the robot observation into checkpoint state space before calling the inner
policy, then adapts the returned policy chunk into absolute robot-joint targets
before the controller sees it. The built-in `PolicyRobotActionAdapter` supplies
the reversible affine joint implementation and supports four declared modes;
the JSON shape is documented in `schema/action_adapter.schema.json`:

| `action_mode` | policy-space absolute target at step `t` |
|---|---|
| `absolute` | `a[t]` |
| `delta_from_observation` | `q_obs + a[t]` |
| `integrated_delta` | `q_obs + cumsum(a)[t]` |
| `velocity` | `q_obs + period_s * cumsum(a)[t]` |

Semantic conversion happens first. The adapter then applies the declared
permutation and affine calibration
`q_robot[r] = scale[r] * q_policy[robot_from_policy[r]] + offset[r]`, and only
then enforces robot position and per-command step limits. The inverse relation
adapts positions and velocities back into policy order. Action/state dimensions,
names, units, camera keys, finite values, observation provenance, and the exact
control period are checked before a target is returned.

The generic adapter intentionally rejects `end_effector` and
`checkpoint_native` frames. Those frames use the declared custom adapter
`module:function`. The included LIBERO boundary loads
`robot.options.kinematics_factory`, whose factory receives `RobotConfig` and
must return an engine with both callables:

```python
encode_policy_state(observation) -> array | RobotState
policy_chunk_to_robot_targets(
    observation, policy_values, period_s, action_adapter_config
) -> array[T, robot_action_dim]  # absolute robot-joint targets
```

The adapter checks required cameras, state/action dimensions, timestamps,
control period, finite values, joint limits, and per-step limits; it never clips
an invalid IK result into range. The SO101 and Dobot LIBERO examples declare
mixed semantics: Cartesian coordinates 0–5 are
`delta_from_observation`, while coordinate 6 is an `absolute` gripper command.
The engine must implement that checkpoint-specific convention, FK/state
encoding, IK, units, and gripper conversion explicitly.

An unvalidated identity remains available for dry-run plumbing. Setting
`robot.dry_run=false` requires `action_adapter.validated=true`, a reviewed
non-placeholder adapter/kinematics factory and calibration ID, complete
names/units/mapping evidence where applicable, and hardware-validated limits.
The example kinematics factories and calibration IDs are deliberate
placeholders, so changing only `dry_run` cannot authorize motion.

## Install

Start from the RLinf embodied environment for OpenPI. Add only the backends used
by the robot:

```bash
pip install -r examples/embodiment/fibocom_vla/requirements/base.txt
pip install -r examples/embodiment/fibocom_vla/requirements/so101.txt
# or
pip install -r examples/embodiment/fibocom_vla/requirements/dobot_nova.txt
```

LeRobot `0.4.4` is pinned because it supports Python 3.10/3.11 used by RLinf
`release/v0.2`; current LeRobot main requires Python 3.12. ROS2 Python packages
come from the selected ROS2 distribution rather than PyPI.

## CPU gates

```bash
python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/mock.json

python -m rlinf.projects.fibocom_vla.cli actor-report \
  --config examples/embodiment/fibocom_vla/config/mock.json

python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json \
  --steps 16

pytest -q tests/unit_tests/projects/fibocom_vla
```

## π0.5 checkpoint dry run and fail-closed assembly

The model path and checkpoint-specific OpenPI data config are deliberately not
embedded in source control. `FIBOCOM_PI05_MODEL_PATH` and
`FIBOCOM_PI05_CONFIG_NAME` are always required; preprocessing semantics are not
guessed. The loaded model's `action_horizon` must exactly match the stack
configuration. RLinf v0.2's stock `pi05_libero` horizon is 10 and is therefore
incompatible with the checked-in 50-step residual examples; `action_chunk`
only truncates output and cannot extend it. Register/select a checkpoint data
config trained for horizon 50 rather than silently overriding the stock
checkpoint. If residual RL is enabled, `FIBOCOM_RESIDUAL_CHECKPOINT` is also
required. If speculative inference is enabled, both
`FIBOCOM_DRAFT_POLICY_FACTORY` and
`FIBOCOM_PARALLEL_VERIFIER_FACTORY` must resolve to compatible factories.
Omitting any environment variable required by the enabled stack fails closed
instead of silently running a different policy.

```bash
export FIBOCOM_PI05_MODEL_PATH=/absolute/path/to/pi05/checkpoint
export FIBOCOM_PI05_CONFIG_NAME=your_registered_pi05_h50_config
export FIBOCOM_PI05_DEVICE=cuda
export FIBOCOM_MAIN_CAMERA=main
export FIBOCOM_WRIST_CAMERA=wrist
export FIBOCOM_RESIDUAL_CHECKPOINT=/absolute/path/to/residual.pt
export FIBOCOM_DRAFT_POLICY_FACTORY=your_package.factories:create_draft
export FIBOCOM_PARALLEL_VERIFIER_FACTORY=your_package.factories:create_verifier

python -m rlinf.projects.fibocom_vla.cli run \
  --config examples/embodiment/fibocom_vla/config/so101_realsense.json \
  --instruction "stack the red block on the blue block" \
  --steps 32
```

The checked-in SO101 file deliberately contains placeholder camera,
calibration, and kinematics values, so the command fails before connecting.
After replacing those placeholders with reviewed integration values while
keeping `dry_run=true`, it connects read-only and does not send joint targets.
Real motion requires all of the following:

1. replace placeholder serial/IP/joint limits with hardware-verified values;
2. replace the `action_adapter` and `kinematics_factory` placeholders with
   checkpoint-specific action/state dimensions, names, units, camera contract,
   reviewed FK/IK, gripper conversion, and a non-placeholder calibration ID;
3. set `robot.dry_run` to `false`;
4. for direct Dobot, configure control request/enable gates and independent
   binary gripper DI feedback; for ROS2, verify the selected service backend,
   acknowledged stop services, command subscriber, and SI joint units;
5. pass `--allow-motion` to a finite run;
6. keep the physical emergency stop reachable.

## Evidence boundary

The commands above cover CPU unit, Mock, and named component scope only. They
do not establish GPU performance, TensorRT equivalence, simulation success, or
physical-robot success; those claims remain gated until dedicated runs produce
reviewable artifacts. In particular:

- historical `48.1572→42.7936 ms` bit-exact evidence belongs to a StarVLA/Qwen
  visual path, not automatically to π0.5;
- `58.0→19.1 ms` and `3.04×` are upstream FLASH results unless reproduced under
  this repository's benchmark protocol;
- RTC `76→97 ms` describes extra model compute in a published setting, while
  `5.6 s→4 ms` can only describe exposed waiting/idle time under a stated clock;
- the currently recovered `112`-trajectory and `75.0%→96.875%` artifacts mix
  IsaacLab/Franka routes and different seeds; they are not same-protocol
  SO101/Nova or physical-robot evidence and cannot support the résumé claim.

The current production-integration boundary is also explicit. Speculative
assembly still requires checkpoint-specific draft and parallel-verifier
factories; no π0.5 draft checkpoint or normalized model-space verifier is
checked in. `ShapeStableCUDAGraph` is a tested capture utility, while the
example benchmark is a synthetic MLP rather than a wired OpenPI visual
subgraph. The TensorRT module is an executor/parity/ablation protocol, not a
π0.5 exporter or engine builder. Those paths remain blocked until the model
manifest, GPU environment, and checkpoint-specific integration are available.
