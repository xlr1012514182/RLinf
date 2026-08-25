Fibocom π0.5 Post-training and Real-time Runtime
================================================

This example provides a frozen-base residual π0.5 post-training path and a
hardware-ready inference runtime for SO101, Dobot Nova, generic ROS2, OpenCV,
and RealSense. It also contains shape-gated CUDA Graph capture, continuous
draft/parallel verification, and asynchronous real-time chunking (RTC).

Implementation and experimental claims are kept separate. Upstream FLASH,
paper, résumé, and historical StarVLA values are not local π0.5 results. The
current commands establish CPU unit, Mock, and named-component scope only; they
do not establish GPU speed, TensorRT equivalence, simulation success, or
physical-robot success. Source revisions and the detailed evidence boundary are
recorded in ``examples/embodiment/fibocom_vla/README.md``.

Residual and policy/robot contracts
-----------------------------------

The 50×7 residual actor with hidden width 562 contains exactly 1,818,542
trainable parameters. The 50×6 actor with hidden width 581 contains exactly
1,819,139. Both use one 2048-D frozen-prefix feature from the same base-model
forward and a bounded residual around the complete reference action chunk.

Both CLI runners install ``PolicyRobotAdapterPolicy``. It converts robot state
into checkpoint state before inference and converts the returned policy chunk
into checked absolute robot-joint targets before execution. The built-in path
supports reversible affine joint calibration. End-effector or
checkpoint-native contracts instead load a custom adapter through
``module:function``.

The LIBERO adapter loads ``robot.options.kinematics_factory``. Its factory
receives ``RobotConfig`` and must return an engine implementing:

.. code-block:: python

   encode_policy_state(observation) -> array_or_robot_state
   policy_chunk_to_robot_targets(
       observation, policy_values, period_s, action_adapter_config
   ) -> absolute_targets  # shape [T, robot_action_dim]

The boundary validates camera keys, dimensions, timestamps, control period,
finite values, joint limits, and per-step limits; an invalid IK result is never
silently clipped. The SO101 and Dobot LIBERO examples use mixed action
semantics: coordinates 0–5 are Cartesian ``delta_from_observation`` and
coordinate 6 is an ``absolute`` gripper command. The reviewed engine must
implement the checkpoint-specific state encoding, FK/IK, units, feasibility,
and gripper conversion.

Hardware and lifecycle gates
----------------------------

SO101 is a five-DOF arm plus gripper, not a six-DOF arm. A six-coordinate
Cartesian delta therefore cannot be relabelled as SO101 joints; the example
remains fail-closed until a reviewed, task-specific engine is supplied.

Dobot sends six arm coordinates in degrees through ``ServoJ`` and may send one
binary gripper coordinate through DO. Real seven-dimensional control requires
independent DI gripper feedback. The direct backend checks controller errors,
collision, documented active-low ``SafetyState`` interlocks, and all four
safety-skin approach-pause fields before commanding motion.

The ROS2 joint backend supports acknowledged stop services through either
``std_srvs/srv/Trigger`` (``service_backend=trigger``) or
``dobot_msgs_v4/srv/Stop`` and ``EmergencyStop``
(``service_backend=dobot_v4``). Real mode preflights the command subscriber and
both services and requires explicit SI joint units. See
``examples/embodiment/fibocom_vla/config/ros2_nova5_arm.json`` for the six-axis
Nova5 arm template.

All cameras in one synchronized read share a global timeout budget and must
pass the camera-skew gate. The joined observation then passes maximum-age and
robot-state/camera-skew gates. Every action send checks both source-observation
age and generation age. Real runs additionally require validated calibration,
hardware limits, ``robot.dry_run=false``, and ``--allow-motion``. Partial
connects roll back; bounded runs select stop or emergency stop and then tear
down planner, cameras, and robot while preserving cleanup failures.

Fail-closed model assembly and RTC
----------------------------------

``FIBOCOM_PI05_MODEL_PATH`` and ``FIBOCOM_PI05_CONFIG_NAME`` are always
required. The loaded model ``action_horizon`` must exactly match the stack.
RLinf v0.2's stock ``pi05_libero`` horizon is 10, so it is rejected by the
checked-in 50-step examples: ``action_chunk`` can truncate but cannot extend a
short model horizon. Select a registered horizon-50 checkpoint config instead
of silently overriding the stock checkpoint. Enabling residual RL additionally
requires
``FIBOCOM_RESIDUAL_CHECKPOINT``. Enabling speculative inference additionally
requires both ``FIBOCOM_DRAFT_POLICY_FACTORY`` and
``FIBOCOM_PARALLEL_VERIFIER_FACTORY``. A missing asset or incompatible factory
raises a configuration error instead of selecting a different policy.

Native OpenPI RTC carries the normalized same-forward action in
``ActionChunk.model_values``. The next denoising request uses that exact lineage
for its hard prefix and overlap objective. Missing lineage fails closed. The
current residual wrapper, speculative wrapper, and custom nonlinear IK/FK path
cannot preserve a safe native path, so the planner uses an explicitly labelled
host output-space fallback; that fallback is not VJP guidance. An in-flight
accelerator forward cannot be safely cancelled: if planner shutdown times out,
restart the owning process before reusing its model or GPU context.

Quick CPU component smoke
-------------------------

.. code-block:: bash

   python -m rlinf.projects.fibocom_vla.cli validate-config \
     --config examples/embodiment/fibocom_vla/config/mock.json
   python -m rlinf.projects.fibocom_vla.cli actor-report \
     --config examples/embodiment/fibocom_vla/config/mock.json
   python -m rlinf.projects.fibocom_vla.cli mock-smoke \
     --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
   pytest -q tests/unit_tests/projects/fibocom_vla
