# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line validation, mock smoke, and finite real-robot runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from .config import StackConfig
from .contracts import ChunkPolicy
from .hardware import (
    PolicyRobotAdapterPolicy,
    create_cameras,
    create_robot,
    load_policy_robot_adapter,
)
from .hardware.cameras import SynchronizedCameraRig
from .policy import MockChunkPolicy
from .runtime import RealtimeController


def _load_policy_factory(path: str):
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("policy factory must use module:function syntax")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _validate(args: argparse.Namespace) -> int:
    config = StackConfig.from_json(args.config)
    _print_json(config.to_dict())
    return 0


def _actor_report(args: argparse.Namespace) -> int:
    config = StackConfig.from_json(args.config)
    from .rl.torch_modules import ResidualActor

    actor = ResidualActor(config.residual_rl)
    _print_json(
        {
            "trainable_parameters": actor.parameter_count,
            "millions_rounded": round(actor.parameter_count / 1_000_000, 2),
            "target": config.residual_rl.target_actor_parameters,
            "tolerance": config.residual_rl.parameter_tolerance,
        }
    )
    return 0


def _verify_checkpoint_assets(args: argparse.Namespace) -> int:
    from .assets import load_and_verify_checkpoint_assets

    repository_root = Path(__file__).resolve().parents[3]
    verified = load_and_verify_checkpoint_assets(
        args.manifest,
        args.root,
        repository_root=repository_root,
    )
    if args.config is not None:
        verified.manifest.validate_stack(StackConfig.from_json(args.config))
    runtime = verified.manifest.runtime
    _print_json(
        {
            "verified": True,
            "root": verified.root,
            "manifest": verified.manifest_path,
            "repository_id": verified.manifest.repository_id,
            "revision": verified.manifest.revision,
            "config_name": runtime.config_name,
            "asset_id": runtime.asset_id,
            "norm_stats": verified.norm_stats_path,
            "action_horizon": runtime.action_horizon,
            "model_action_dim": runtime.model_action_dim,
            "environment_action_dim": runtime.environment_action_dim,
            "raw_state_dim": runtime.state_dim,
            "model_state_dim": runtime.model_state_dim,
            "camera_keys": runtime.cameras.observation_keys,
            "file_count": len(verified.manifest.files),
            "hash_mode": "sha256_all_declared_files",
        }
    )
    return 0


def _verify_draft_asset(args: argparse.Namespace) -> int:
    from .draft_assets import load_draft_asset_registry

    registry = load_draft_asset_registry(args.registry)
    verified = registry.verify_asset(args.suite, args.root)
    _print_json(
        {
            "verified": True,
            "suite": verified.asset.suite,
            "path": verified.path,
            "size": verified.asset.size,
            "sha256": verified.asset.sha256,
            "model_repository": registry.model_repository,
            "model_revision": registry.model_revision,
            "code_revision": registry.code_revision,
            "published_base_contract": asdict(registry.base_contract),
            "production_compatibility": "pi0_libero_exact_contract_only",
            "pi05_direct_use": False,
            "pi05_allowed_use": "verified_initialization_only_before_retraining",
        }
    )
    return 0


def _mock_smoke(args: argparse.Namespace) -> int:
    config = StackConfig.from_json(args.config)
    if config.robot.backend != "mock" or any(
        camera.backend != "mock" for camera in config.cameras
    ):
        raise ValueError("mock-smoke requires mock robot and camera backends")
    robot = create_robot(config.robot)
    camera_rig = SynchronizedCameraRig(create_cameras(config.cameras))
    inner_policy = MockChunkPolicy(
        horizon=config.residual_rl.action_horizon,
        action_dim=config.robot.action_adapter.resolved_policy_dim(
            config.robot.action_dim
        ),
        period_s=1.0 / config.robot.control_hz,
    )
    policy = PolicyRobotAdapterPolicy(inner_policy, config.robot)
    result = RealtimeController(
        robot,
        camera_rig,
        policy,
        config.rtc,
        instruction=args.instruction,
    ).run(maximum_control_steps=args.steps)
    _print_json(asdict(result))
    return 0


def _openpi_checkpoint_smoke(args: argparse.Namespace) -> int:
    """Load the manifest-bound checkpoint and run one synthetic H=50 forward."""

    import numpy as np
    import torch

    from .contracts import Observation, RobotState
    from .factories import create_openpi_policy_from_env

    config = StackConfig.from_json(args.config)
    if config.residual_rl.enabled or config.speculative.enabled:
        raise ValueError(
            "openpi-checkpoint-smoke requires residual_rl/speculative disabled"
        )
    policy = create_openpi_policy_from_env(config)
    model = getattr(policy, "model", None)
    if model is None:
        raise TypeError("checkpoint smoke requires the base RLinf OpenPI policy")
    identity = getattr(policy, "fibocom_checkpoint_identity", None)
    if not isinstance(identity, dict):
        raise RuntimeError("checkpoint smoke requires a verified asset manifest")

    timestamp_ns = time.monotonic_ns()
    state = np.zeros(config.robot.action_dim, dtype=np.float32)
    for index, (lower, upper) in enumerate(
        zip(config.robot.joint_lower, config.robot.joint_upper, strict=True)
    ):
        if not lower <= 0 <= upper:
            state[index] = np.float32((lower + upper) / 2)
    images = {
        camera.name: np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        for camera in config.cameras
    }
    observation = Observation(
        state=RobotState(
            joint_positions=state,
            joint_names=config.robot.joint_names,
            timestamp_ns=timestamp_ns,
        ),
        images=images,
        instruction=args.instruction,
        timestamp_ns=timestamp_ns,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = policy.predict(observation)
    action = np.ascontiguousarray(output.action.values, dtype=np.float32)
    model_actions = output.action.model_values
    if model_actions is None:
        raise RuntimeError("OpenPI smoke output lost normalized model actions")
    model_actions = np.ascontiguousarray(model_actions, dtype=np.float32)
    load_report = getattr(model, "_rlinf_checkpoint_load_report", None)
    if not isinstance(load_report, dict):
        raise RuntimeError("loaded model lacks its checkpoint compatibility report")
    if load_report.get("missing_keys") or load_report.get("unexpected_keys"):
        raise RuntimeError("checkpoint smoke observed incompatible state-dict keys")
    if action.shape != (
        config.residual_rl.action_horizon,
        config.residual_rl.action_dim,
    ):
        raise RuntimeError(f"unexpected environment action shape: {action.shape}")
    if model_actions.shape != (
        identity["action_horizon"],
        identity["model_action_dim"],
    ):
        raise RuntimeError(f"unexpected model action shape: {model_actions.shape}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _print_json(
        {
            "smoke_passed": True,
            "validation_scope": "one_synthetic_observation_not_a_benchmark",
            "checkpoint_identity": identity,
            "load_report": {
                "source_kind": load_report.get("source_kind"),
                "selected_paths": load_report.get("selected_paths"),
                "missing_keys": load_report.get("missing_keys"),
                "unexpected_keys": load_report.get("unexpected_keys"),
                "data_asset_id": load_report.get("data_asset_id"),
                "use_quantile_norm": load_report.get("use_quantile_norm"),
            },
            "parameter_count": parameter_count,
            "environment_action_shape": action.shape,
            "model_action_shape": model_actions.shape,
            "environment_action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
            "model_action_sha256": hashlib.sha256(model_actions.tobytes()).hexdigest(),
            "all_outputs_finite": bool(
                np.isfinite(action).all() and np.isfinite(model_actions).all()
            ),
            "policy_path": output.path,
            "stack_layers": getattr(policy, "fibocom_stack_layers", ()),
            "seed": args.seed,
        }
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    config = StackConfig.from_json(args.config)
    if not config.robot.dry_run and not args.allow_motion:
        raise PermissionError(
            "real motion requires both robot.dry_run=false and --allow-motion"
        )
    if not config.robot.dry_run and bool(
        config.robot.options.get("limits_require_hardware_validation", True)
    ):
        raise PermissionError(
            "real motion is blocked until joint limits are hardware-validated and "
            "robot.options.limits_require_hardware_validation=false"
        )
    adapter = load_policy_robot_adapter(config.robot)
    factory = _load_policy_factory(args.policy_factory)
    inner_policy = factory(config)
    if not isinstance(inner_policy, ChunkPolicy):
        raise TypeError("policy factory did not return a ChunkPolicy-compatible object")
    policy = PolicyRobotAdapterPolicy(inner_policy, config.robot, adapter=adapter)
    robot = create_robot(config.robot)
    camera_rig = SynchronizedCameraRig(
        create_cameras(config.cameras),
        maximum_skew_ms=args.maximum_camera_skew_ms,
    )
    result = RealtimeController(
        robot,
        camera_rig,
        policy,
        config.rtc,
        instruction=args.instruction,
    ).run(maximum_control_steps=args.steps)
    _print_json(asdict(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser without importing robot/model dependencies."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    validate.set_defaults(handler=_validate)

    actor = subparsers.add_parser("actor-report")
    actor.add_argument("--config", type=Path, required=True)
    actor.set_defaults(handler=_actor_report)

    assets = subparsers.add_parser("verify-checkpoint-assets")
    assets.add_argument("--manifest", type=Path, required=True)
    assets.add_argument("--root", type=Path, required=True)
    assets.add_argument("--config", type=Path)
    assets.set_defaults(handler=_verify_checkpoint_assets)

    draft = subparsers.add_parser("verify-draft-asset")
    draft.add_argument("--registry", type=Path, required=True)
    draft.add_argument("--root", type=Path, required=True)
    draft.add_argument(
        "--suite",
        choices=("libero_10", "libero_goal", "libero_object", "libero_spatial"),
        required=True,
    )
    draft.set_defaults(handler=_verify_draft_asset)

    mock = subparsers.add_parser("mock-smoke")
    mock.add_argument("--config", type=Path, required=True)
    mock.add_argument("--instruction", default="stack the block")
    mock.add_argument("--steps", type=int, default=16)
    mock.set_defaults(handler=_mock_smoke)

    checkpoint_smoke = subparsers.add_parser("openpi-checkpoint-smoke")
    checkpoint_smoke.add_argument("--config", type=Path, required=True)
    checkpoint_smoke.add_argument(
        "--instruction", default="stack the red block on the blue block"
    )
    checkpoint_smoke.add_argument("--seed", type=int, default=17)
    checkpoint_smoke.set_defaults(handler=_openpi_checkpoint_smoke)

    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--policy-factory",
        default=("rlinf.projects.fibocom_vla.factories:create_openpi_policy_from_env"),
    )
    run.add_argument("--instruction", required=True)
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--maximum-camera-skew-ms", type=float, default=35.0)
    run.add_argument("--allow-motion", action="store_true")
    run.set_defaults(handler=_run)
    return parser


def main() -> int:
    """Run the selected finite command."""

    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
