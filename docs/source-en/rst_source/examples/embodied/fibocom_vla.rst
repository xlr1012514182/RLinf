Fibocom π0.5 VLA Stack
================================

This example connects a frozen π0.5 policy to residual post-training,
continuous-action inference and robot interfaces. The source distribution
contains runtime contracts, configuration templates, tests and benchmark
programs. Weights, task data, deployment calibration and run outputs are
provided separately.

Architecture
------------

* Frozen base, bounded residual, and approved Draft/verifier are alternative
  policy profiles. Residual and speculative modes are mutually exclusive.
* The asynchronous planner overlaps action-chunk generation and execution.
  Native RTC requires preserved model-space lineage; wrapped or nonlinear
  retargeting paths use an explicit host-space fallback.
* Visual CUDA Graph capture covers the tower/projector and falls back to eager
  on ineligible inputs. RTC, speculative and gradient contexts bypass capture.
* TensorRT utilities require an externally supplied exported graph/network;
  the repository does not bundle a whole-policy π0.5 exporter or engine.

CPU quick start
---------------

From the repository root on Ubuntu/WSL2 x86_64:

.. code-block:: bash

   bash requirements/fibocom_vla_quickstart.sh
   .venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
     --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
   .venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
     --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
   .venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla

This path checks CPU components and a Mock loop, without weights or hardware.
Use the separate RLinf OpenPI environment for real-model execution.

Model assembly
--------------

The main asset is ``RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle`` at revision
``fa8df6ed103db0f5549c122f3a17c00ba6426c98``. The registered
``pi05_aloha_robotwin`` runtime uses H=50, raw state/action 14, model
state/action 32, and ordered high/left-wrist/right-wrist cameras. The publisher
H=10 config remains unchanged; a separate manifest binds the runtime and
transform source fingerprints.

Bind ``FIBOCOM_PI05_MODEL_PATH`` and ``FIBOCOM_PI05_ASSET_MANIFEST`` after
asset verification. An explicit ``FIBOCOM_PI05_CONFIG_NAME`` must match the
manifest. Residual assembly additionally needs ``FIBOCOM_RESIDUAL_CHECKPOINT``.
Speculative assembly needs ``FIBOCOM_PI05_DRAFT_MANIFEST``,
``FIBOCOM_PI05_DRAFT_EVALUATION``, ``FIBOCOM_PI05_DRAFT_APPROVAL``,
``FIBOCOM_PI05_DRAFT_PUBLIC_KEY`` and ``FIBOCOM_PI05_DRAFT_SIGNING_KEY_ID``;
the factory builds the bound verifier. Published π0 LIBERO Drafts are only
warm-start assets for this π0.5 target.

Training and hardware
---------------------

The residual initialization/update CLI consumes external rollout archives;
task collection and evaluation are integration responsibilities. The Draft
CLI materializes teacher caches, trains and exports unsigned candidates;
held-out evaluation and independent approval remain separate.

Robot interfaces cover SO101, Dobot Nova TCP and ROS2 joints, with OpenCV,
RealSense and ROS2 cameras. Real templates remain dry-run or motion-locked.
Dual-Aloha H50×14 requires a task-specific retargeting engine for a single arm,
with reviewed calibration, units, limits, feedback freshness and stop behavior.
Motion additionally requires ``dry_run=false`` and ``--allow-motion``;
config validation alone is not motion authorization.

Further reading
---------------

* :download:`Implementation guide <../../../../../examples/embodiment/fibocom_vla/README.md>`
* :download:`Validation procedures <../../../../fibocom_vla/VALIDATION.md>`
* :download:`Source inventory and licensing <../../../../fibocom_vla/SOURCES.md>`

This is an independent adaptation, not an official RLinf or Fibocom release.
