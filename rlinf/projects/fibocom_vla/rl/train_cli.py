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

"""Initialize and run one auditable residual PPO/twin-Q update."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import StackConfig
from .archive import load_rollout_archive
from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    file_sha256,
    load_residual_checkpoint,
    save_residual_checkpoint,
)
from .torch_modules import ResidualActor, StateValueCritic, TwinQCritic
from .trainer import HybridResidualPPOTrainer


def _git_revision() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return revision, dirty


def _seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_trainer(
    config: StackConfig, device: torch.device
) -> HybridResidualPPOTrainer:
    actor = ResidualActor(config.residual_rl).to(device)
    critic = TwinQCritic(config.residual_rl).to(device)
    value = StateValueCritic(config.residual_rl).to(device)
    return HybridResidualPPOTrainer(
        actor,
        critic,
        config.residual_rl,
        value_critic=value,
    )


def _checkpoint_payload(
    trainer: HybridResidualPPOTrainer,
    *,
    config_sha256: str,
    base_model: str,
    base_model_revision: str,
    seed: int,
    parent_checkpoint_sha256: str | None,
    rollout_sha256: str | None,
    update_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    revision, dirty = _git_revision()
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": revision,
            "git_dirty": dirty,
            "configuration_sha256": config_sha256,
            "base_model": base_model,
            "base_model_revision": base_model_revision,
            "seed": seed,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "rollout_sha256": rollout_sha256,
            "declared_ppo_epochs": trainer.config.ppo_epochs,
            "update_metrics": update_metrics,
            "parameter_counts": {
                "residual_actor": trainer.actor.parameter_count,
                "twin_q": sum(
                    parameter.numel() for parameter in trainer.critic.parameters()
                ),
                "state_value": sum(
                    parameter.numel() for parameter in trainer.value_critic.parameters()
                ),
            },
        },
        "trainer_state": trainer.state_dict(),
    }


def initialize_checkpoint(
    config_path: Path,
    output_path: Path,
    *,
    base_model: str,
    base_model_revision: str,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Create the exact behavior checkpoint that must collect rollouts."""

    if not base_model.strip() or not base_model_revision.strip():
        raise ValueError("base model identity and revision are required")
    config = StackConfig.from_json(config_path)
    _seed_everything(seed)
    trainer = _create_trainer(config, torch.device(device))
    config_digest = file_sha256(config_path)
    payload = _checkpoint_payload(
        trainer,
        config_sha256=config_digest,
        base_model=base_model,
        base_model_revision=base_model_revision,
        seed=seed,
        parent_checkpoint_sha256=None,
        rollout_sha256=None,
        update_metrics=None,
    )
    digest = save_residual_checkpoint(output_path, payload)
    return {
        "checkpoint": str(output_path),
        "checkpoint_sha256": digest,
        "configuration_sha256": config_digest,
        "residual_actor_parameters": trainer.actor.parameter_count,
    }


def update_checkpoint(
    config_path: Path,
    input_checkpoint: Path,
    rollout_path: Path,
    output_checkpoint: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Run the declared single PPO epoch after all lineage gates pass."""

    config = StackConfig.from_json(config_path)
    target_device = torch.device(device)
    checkpoint, checkpoint_digest = load_residual_checkpoint(
        input_checkpoint, map_location=target_device
    )
    rollout, rollout_metadata, _manifest = load_rollout_archive(rollout_path)
    config_digest = file_sha256(config_path)
    checkpoint_metadata = checkpoint["metadata"]
    if checkpoint_metadata.get("configuration_sha256") != config_digest:
        raise ValueError("checkpoint configuration SHA-256 does not match --config")
    if rollout_metadata.configuration_sha256 != config_digest:
        raise ValueError("rollout configuration SHA-256 does not match --config")
    if rollout_metadata.behavior_policy_revision != checkpoint_digest:
        raise ValueError(
            "rollout behavior_policy_revision is not the input checkpoint SHA-256"
        )
    for key in ("base_model", "base_model_revision"):
        if getattr(rollout_metadata, key) != checkpoint_metadata.get(key):
            raise ValueError(f"rollout and checkpoint disagree on {key}")
    expected_geometry = (
        config.residual_rl.feature_dim,
        config.residual_rl.action_horizon,
        config.residual_rl.action_dim,
    )
    observed_geometry = (
        rollout.features.shape[-1],
        rollout.reference_actions.shape[-2],
        rollout.reference_actions.shape[-1],
    )
    if observed_geometry != expected_geometry:
        raise ValueError(
            f"rollout geometry {observed_geometry} != config {expected_geometry}"
        )

    _seed_everything(int(checkpoint_metadata["seed"]))
    trainer = _create_trainer(config, target_device)
    trainer.load_state_dict(checkpoint["trainer_state"])
    metrics = trainer.update(rollout)
    rollout_digest = file_sha256(rollout_path)
    payload = _checkpoint_payload(
        trainer,
        config_sha256=config_digest,
        base_model=rollout_metadata.base_model,
        base_model_revision=rollout_metadata.base_model_revision,
        seed=int(checkpoint_metadata["seed"]),
        parent_checkpoint_sha256=checkpoint_digest,
        rollout_sha256=rollout_digest,
        update_metrics=asdict(metrics),
    )
    output_digest = save_residual_checkpoint(output_checkpoint, payload)
    return {
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": output_digest,
        "parent_checkpoint_sha256": checkpoint_digest,
        "rollout_sha256": rollout_digest,
        "metrics": asdict(metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded initialization/update command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--config", type=Path, required=True)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--base-model", required=True)
    initialize.add_argument("--base-model-revision", required=True)
    initialize.add_argument("--seed", type=int, default=17)
    initialize.add_argument("--device", default="cpu")

    update = subparsers.add_parser("update")
    update.add_argument("--config", type=Path, required=True)
    update.add_argument("--input-checkpoint", type=Path, required=True)
    update.add_argument("--rollout", type=Path, required=True)
    update.add_argument("--output", type=Path, required=True)
    update.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    """Run one selected residual-training operation."""

    args = build_parser().parse_args()
    if args.command == "initialize":
        result = initialize_checkpoint(
            args.config,
            args.output,
            base_model=args.base_model,
            base_model_revision=args.base_model_revision,
            seed=args.seed,
            device=args.device,
        )
    else:
        result = update_checkpoint(
            args.config,
            args.input_checkpoint,
            args.rollout,
            args.output,
            device=args.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
