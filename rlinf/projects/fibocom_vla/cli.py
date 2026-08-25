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
import importlib
import json
import logging
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

    mock = subparsers.add_parser("mock-smoke")
    mock.add_argument("--config", type=Path, required=True)
    mock.add_argument("--instruction", default="stack the block")
    mock.add_argument("--steps", type=int, default=16)
    mock.set_defaults(handler=_mock_smoke)

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
