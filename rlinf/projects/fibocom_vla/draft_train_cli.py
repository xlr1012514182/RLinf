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

"""Standalone, non-signing CLI for pi0.5 Draft cache/train/export stages."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .draft_assets import load_draft_asset_registry
from .errors import ConfigurationError
from .inference.draft_head import load_draft_chunk_head
from .inference.pi05_draft import derive_pi05_draft_from_warm_start
from .inference.pi05_draft_training import (
    DraftTrainingConfig,
    OpenPITeacherOwner,
    create_teacher_owner_from_env,
    current_git_revision,
    export_candidate_from_run,
    load_source_adapter,
    materialize_teacher_cache,
    train_draft_model,
    verify_teacher_cache,
    verify_training_run,
)


def _json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"failed to read JSON object {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return dict(value)


def _load_teacher_owner(
    specification: str, options: Mapping[str, Any]
) -> OpenPITeacherOwner:
    if specification.count(":") != 1:
        raise ConfigurationError(
            "teacher owner factory must use module:function syntax"
        )
    module_name, function_name = specification.split(":", 1)
    if not module_name.strip() or not function_name.strip():
        raise ConfigurationError(
            "teacher owner factory must use module:function syntax"
        )
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ConfigurationError("teacher owner factory is not callable")
    owner = factory(options=dict(options))
    if type(owner) is not OpenPITeacherOwner:
        raise ConfigurationError(
            "teacher owner factory must return OpenPITeacherOwner; precomputed "
            "teacher actions are not accepted"
        )
    return owner


def _derive_head(args: argparse.Namespace, *, target: Any) -> Any:
    registry = load_draft_asset_registry(args.draft_registry)
    resolution = registry.resolve_warm_start(
        args.draft_suite,
        args.draft_asset_root,
        registry.base_contract,
    )
    dtype = (
        torch.bfloat16 if target.draft_parameter_dtype == "bfloat16" else torch.float32
    )
    warm_start = load_draft_chunk_head(
        resolution,
        purpose="warm_start",
        device=args.device,
        dtype=dtype,
    )
    head, _report = derive_pi05_draft_from_warm_start(
        warm_start,
        target_contract=target,
        initialization_seed=args.initialization_seed,
        device=args.device,
    )
    return head


def _add_warm_start_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--draft-registry", type=Path, required=True)
    parser.add_argument("--draft-asset-root", type=Path, required=True)
    parser.add_argument(
        "--draft-suite",
        choices=("libero_10", "libero_goal", "libero_object", "libero_spatial"),
        required=True,
    )
    parser.add_argument("--initialization-seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded cache/training/candidate command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--source-adapter", required=True)
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--source-manifest", type=Path, required=True)
    materialize.add_argument("--source-options", type=Path)
    teacher = materialize.add_mutually_exclusive_group(required=True)
    teacher.add_argument("--stack-config", type=Path)
    teacher.add_argument("--teacher-owner-factory")
    materialize.add_argument("--teacher-options", type=Path)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--split-seed", type=int, default=17)
    materialize.add_argument("--validation-fraction", type=float, default=0.2)

    verify_cache = commands.add_parser("verify-cache")
    verify_cache.add_argument("--manifest", type=Path, required=True)

    train = commands.add_parser("train")
    train.add_argument("--cache-manifest", type=Path, required=True)
    train.add_argument("--training-config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume-run-manifest", type=Path)
    train.add_argument("--stop-after-new-steps", type=int)
    _add_warm_start_arguments(train)

    verify_run = commands.add_parser("verify-run")
    verify_run.add_argument("--manifest", type=Path, required=True)

    export = commands.add_parser("export-candidate")
    export.add_argument("--cache-manifest", type=Path, required=True)
    export.add_argument("--run-manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    _add_warm_start_arguments(export)
    return parser


def _result(**values: Any) -> None:
    print(json.dumps(values, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    """Execute one selected stage without any production-approval capability."""

    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        source_options = _json_object(args.source_options)
        teacher_options = _json_object(args.teacher_options)
        source = load_source_adapter(
            args.source_adapter,
            source_root=args.source_root,
            source_manifest_path=args.source_manifest,
            options=source_options,
        )
        if args.stack_config is not None:
            if args.teacher_options is not None:
                raise ConfigurationError(
                    "--teacher-options is only valid with --teacher-owner-factory"
                )
            owner = create_teacher_owner_from_env(
                {"stack_config": str(args.stack_config.expanduser().resolve())}
            )
        else:
            owner = _load_teacher_owner(args.teacher_owner_factory, teacher_options)
        cache = materialize_teacher_cache(
            source,
            source_manifest_path=args.source_manifest,
            teacher_owner=owner,
            output_dir=args.output,
            split_seed=args.split_seed,
            validation_fraction=args.validation_fraction,
        )
        _result(
            cache_manifest=str(cache.manifest_path),
            cache_manifest_sha256=cache.manifest_sha256,
            sample_count=cache.sample_count,
            production_authorized=False,
        )
        return 0
    if args.command == "verify-cache":
        cache = verify_teacher_cache(args.manifest)
        _result(
            cache_manifest=str(cache.manifest_path),
            cache_manifest_sha256=cache.manifest_sha256,
            sample_count=cache.sample_count,
        )
        return 0
    if args.command == "verify-run":
        run = verify_training_run(args.manifest)
        _result(
            run_manifest=str(run.manifest_path),
            run_manifest_sha256=run.manifest_sha256,
            optimizer_steps=run.optimizer_steps,
            complete=run.complete,
            production_authorized=False,
        )
        return 0

    cache = verify_teacher_cache(args.cache_manifest)
    head = _derive_head(args, target=cache.target_contract)
    if args.command == "train":
        config = DraftTrainingConfig.from_dict(_json_object(args.training_config))
        revision, dirty = current_git_revision()
        if dirty or revision != config.training_code_revision:
            raise ConfigurationError(
                "training config must bind the current clean Git revision"
            )
        resume = (
            None
            if args.resume_run_manifest is None
            else verify_training_run(args.resume_run_manifest)
        )
        run = train_draft_model(
            cache,
            head,
            config,
            output_dir=args.output,
            resume_run=resume,
            stop_after_new_steps=args.stop_after_new_steps,
        )
        _result(
            run_manifest=str(run.manifest_path),
            run_manifest_sha256=run.manifest_sha256,
            optimizer_steps=run.optimizer_steps,
            complete=run.complete,
            metrics=dict(run.manifest["metrics"]),
            production_authorized=False,
        )
        return 0
    if args.command == "export-candidate":
        run = verify_training_run(args.run_manifest)
        artifact = export_candidate_from_run(
            run,
            cache,
            head,
            output_dir=args.output,
        )
        _result(
            candidate_manifest=str(artifact.manifest_path),
            candidate_manifest_sha256=artifact.manifest_sha256,
            weights_sha256=artifact.weights_sha256,
            production_authorized=False,
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
