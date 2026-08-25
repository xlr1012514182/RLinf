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

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rlinf.projects.fibocom_vla.assets import (
    CheckpointAssetManifest,
    VerifiedCheckpointAssets,
    load_and_verify_checkpoint_assets,
    require_verified_checkpoint_assets,
)
from rlinf.projects.fibocom_vla.config import (
    ActionAdapterConfig,
    ResidualRLConfig,
    RobotConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "checkpoint"
    root.mkdir()
    config_bytes = json.dumps(
        {
            "action_dim": 32,
            "action_horizon": 10,
            "precision": "bfloat16",
        }
    ).encode()
    norm_bytes = json.dumps(
        {
            "norm_stats": {
                key: {
                    "mean": [0.0] * 7,
                    "std": [1.0] * 7,
                    "q01": [-1.0] * 7,
                    "q99": [1.0] * 7,
                }
                for key in ("state", "actions")
            }
        }
    ).encode()
    weights = b"source-locked-test-weights"
    index_bytes = json.dumps(
        {
            "metadata": {"total_size": len(weights)},
            "weight_map": {"model.weight": "weights.safetensors"},
        }
    ).encode()
    (root / "config.json").write_bytes(config_bytes)
    (root / "assets").mkdir()
    (root / "assets" / "norm_stats.json").write_bytes(norm_bytes)
    (root / "weights.safetensors").write_bytes(weights)
    (root / "model.safetensors.index.json").write_bytes(index_bytes)

    repository = tmp_path / "repository"
    repository.mkdir()
    source_bytes = b"line_one\r\nline_two\r\n"
    (repository / "runtime_transform.py").write_bytes(source_bytes)
    source_digest = _digest(source_bytes.replace(b"\r\n", b"\n"))
    files = (
        ("config.json", config_bytes, "publisher_config"),
        ("assets/norm_stats.json", norm_bytes, "selected_norm_stats"),
        ("weights.safetensors", weights, "weights_shard"),
        ("model.safetensors.index.json", index_bytes, "weights_index"),
    )
    state_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
        "aux",
    )
    manifest = {
        "schema_version": 1,
        "identity": {
            "repository_id": "unit/test-pi05",
            "repository_url": "https://example.invalid/unit/test-pi05",
            "revision": "a" * 40,
        },
        "runtime": {
            "model_family": "pi05",
            "config_name": "unit_test_pi05",
            "asset_id": "unit/test",
            "action_horizon": 50,
            "model_action_dim": 32,
            "max_token_length": 200,
            "discrete_state_input": True,
            "environment_action_dim": 7,
            "state_dim": 7,
            "model_state_dim": 32,
            "state_semantics": list(state_names),
            "environment_action_semantics": list(state_names),
            "delta_action_mask": [False] * 7,
            "normalization": "quantile_q01_q99",
            "norm_stats_path": "assets/norm_stats.json",
            "checkpoint_config_path": "config.json",
            "action_mode": "absolute",
            "policy_frame": "joint",
            "extra_delta_transform": False,
            "adapt_to_pi": False,
            "cameras": {
                "main": "main",
                "wrists": [],
                "model_image_keys": ["base_0_rgb"],
            },
            "source_fingerprints": [
                {
                    "path": "runtime_transform.py",
                    "sha256_lf": source_digest,
                    "repository": "https://example.invalid/repository",
                    "revision": "b" * 40,
                }
            ],
        },
        "publisher_config": {
            "action_dim": 32,
            "action_horizon": 10,
            "precision": "bfloat16",
        },
        "files": [
            {
                "path": path,
                "size": len(data),
                "sha256": _digest(data),
                "role": role,
            }
            for path, data, role in files
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, root, repository


def test_manifest_verifies_checkpoint_stats_and_canonical_source(tmp_path: Path):
    manifest_path, root, repository = _write_fixture(tmp_path)

    verified = load_and_verify_checkpoint_assets(
        manifest_path,
        root,
        repository_root=repository,
    )

    assert verified.root == root.resolve()
    assert verified.norm_stats_path == (root / "assets" / "norm_stats.json")
    assert verified.manifest.runtime.action_horizon == 50
    assert verified.checkpoint_id == f"unit/test-pi05@{'a' * 40}"
    assert require_verified_checkpoint_assets(verified) is verified
    with pytest.raises(FrozenInstanceError):
        verified.root = tmp_path  # type: ignore[misc]
    verified.manifest.validate_stack(StackConfig())


def test_manual_verified_result_has_no_verification_capability(tmp_path: Path):
    manifest_path, root, _ = _write_fixture(tmp_path)
    manifest = CheckpointAssetManifest.from_path(manifest_path)
    forged = VerifiedCheckpointAssets(
        manifest=manifest,
        root=root.resolve(),
        manifest_path=manifest_path.resolve(),
        _verification_proof=object(),
    )

    with pytest.raises(ConfigurationError, match="verification proof"):
        require_verified_checkpoint_assets(forged)

    with pytest.raises(ConfigurationError, match="verification proof"):
        require_verified_checkpoint_assets(object())


def test_manifest_requires_repository_root_for_source_proof(tmp_path: Path):
    manifest_path, root, _ = _write_fixture(tmp_path)

    with pytest.raises(ConfigurationError, match="repository_root is required"):
        load_and_verify_checkpoint_assets(manifest_path, root)


def test_manifest_detects_tampered_weight_before_model_load(tmp_path: Path):
    manifest_path, root, repository = _write_fixture(tmp_path)
    (root / "weights.safetensors").write_bytes(b"tampered-test-weights-data")

    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        load_and_verify_checkpoint_assets(
            manifest_path,
            root,
            repository_root=repository,
        )


def test_manifest_rejects_stack_semantic_mismatch(tmp_path: Path):
    manifest_path, _, _ = _write_fixture(tmp_path)
    manifest = CheckpointAssetManifest.from_path(manifest_path)
    config = StackConfig(
        residual_rl=ResidualRLConfig(enabled=False, action_dim=6),
    )

    with pytest.raises(ConfigurationError, match=r"action_dim stack=6 asset=7"):
        manifest.validate_stack(config)


def test_manifest_rejects_named_semantic_and_coordinate_mismatch(tmp_path: Path):
    manifest_path, _, _ = _write_fixture(tmp_path)
    manifest = CheckpointAssetManifest.from_path(manifest_path)
    joint_names = StackConfig().robot.joint_names
    config = StackConfig(
        robot=RobotConfig(
            action_adapter=ActionAdapterConfig(
                policy_names=joint_names,
                policy_state_names=("wrong_joint", *joint_names[1:]),
                coordinate_modes=("velocity",) * 7,
            )
        )
    )

    with pytest.raises(
        ConfigurationError,
        match="policy_state_names.*coordinate_modes",
    ):
        manifest.validate_stack(config)


@pytest.mark.parametrize(
    "relative_path",
    (
        "model_state_dict/full_weights.pt",
        "actor/model_state_dict/full_weights.pt",
    ),
)
def test_manifest_rejects_loader_shadow_paths(tmp_path: Path, relative_path: str):
    manifest_path, root, repository = _write_fixture(tmp_path)
    shadow = root / relative_path
    shadow.parent.mkdir(parents=True)
    shadow.write_bytes(b"unverified-shadow")

    with pytest.raises(ConfigurationError, match="would shadow"):
        load_and_verify_checkpoint_assets(
            manifest_path,
            root,
            repository_root=repository,
        )


def test_manifest_rejects_extra_checkpoint_root_safetensors(tmp_path: Path):
    manifest_path, root, repository = _write_fixture(tmp_path)
    (root / "unverified.safetensors").write_bytes(b"extra")

    with pytest.raises(ConfigurationError, match="safetensors set differs"):
        load_and_verify_checkpoint_assets(
            manifest_path,
            root,
            repository_root=repository,
        )


def test_manifest_rejects_weight_index_shard_mismatch(tmp_path: Path):
    manifest_path, root, repository = _write_fixture(tmp_path)
    invalid_index = json.dumps(
        {"weight_map": {"model.weight": "other.safetensors"}}
    ).encode()
    (root / "model.safetensors.index.json").write_bytes(invalid_index)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_entry = next(
        entry for entry in data["files"] if entry["role"] == "weights_index"
    )
    index_entry["size"] = len(invalid_index)
    index_entry["sha256"] = _digest(invalid_index)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="index shard set differs"):
        load_and_verify_checkpoint_assets(
            manifest_path,
            root,
            repository_root=repository,
        )


def test_verified_assets_cannot_disable_sha256(tmp_path: Path):
    manifest_path, root, repository = _write_fixture(tmp_path)

    with pytest.raises(ConfigurationError, match="cannot be disabled"):
        load_and_verify_checkpoint_assets(
            manifest_path,
            root,
            repository_root=repository,
            verify_hashes=False,
        )


def test_manifest_rejects_runtime_file_with_wrong_role(tmp_path: Path):
    manifest_path, _, _ = _write_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["files"][1]["role"] = "unselected_norm_stats"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="selected_norm_stats"):
        CheckpointAssetManifest.from_path(manifest_path)


def test_manifest_rejects_delta_mask_that_conflicts_with_transform(tmp_path: Path):
    manifest_path, _, _ = _write_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["runtime"]["delta_action_mask"][0] = True
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must be all false"):
        CheckpointAssetManifest.from_path(manifest_path)


def test_manifest_rejects_path_traversal(tmp_path: Path):
    manifest_path, _, _ = _write_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["files"][0]["path"] = "../config.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="stay inside"):
        CheckpointAssetManifest.from_path(manifest_path)


def test_verified_openpi_target_is_derived_from_asset_and_loader_proofs(
    tmp_path: Path,
):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from rlinf.projects.fibocom_vla.inference.openpi_speculative import (
        OpenPIParallelVerifier,
        VerifiedOpenPITarget,
    )

    manifest_path, root, repository = _write_fixture(tmp_path)
    verified = load_and_verify_checkpoint_assets(
        manifest_path,
        root,
        repository_root=repository,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(
                config_name="unit_test_pi05",
                action_horizon=50,
                action_chunk=50,
                action_dim=32,
                action_env_dim=7,
                max_token_len=200,
                num_images_in_input=1,
                pi05=True,
            )
            self._rlinf_checkpoint_load_report = {
                "source_kind": "safetensors_shards",
                "missing_keys": (),
                "unexpected_keys": (),
                "selected_paths": tuple(str(path) for path in verified.weight_paths),
            }

    model = _Model()
    target = VerifiedOpenPITarget.from_verified_assets(model, verified)
    verifier = OpenPIParallelVerifier(target)

    assert target.contract.policy_family == "pi0.5"
    assert target.contract.action_horizon == 50
    assert target.contract.model_action_dim == 32
    assert target.contract.env_action_indices == tuple(range(7))
    assert target.contract.camera_keys == ("main",)
    assert target.contract.model_camera_keys == ("base_0_rgb",)
    assert "weights-sha256:" in target.contract.checkpoint_id
    assert verifier.verified_target is target
    assert model.training is False
    assert model.weight.requires_grad is False

    with pytest.raises(ConfigurationError, match="issued from verified"):
        VerifiedOpenPITarget(
            model=model,
            assets=verified,
            contract=target.contract,
            _target_token=object(),
        )
