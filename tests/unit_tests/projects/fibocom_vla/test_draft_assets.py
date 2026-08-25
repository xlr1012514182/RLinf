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
from dataclasses import replace
from pathlib import Path

import pytest

from rlinf.projects.fibocom_vla.draft_assets import (
    PUBLISHED_CODE_REVISION,
    PUBLISHED_MODEL_REVISION,
    DraftAsset,
    DraftAssetRegistry,
    DraftTargetContract,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_REGISTRY_PATH = (
    _PROJECT_ROOT
    / "examples"
    / "embodiment"
    / "fibocom_vla"
    / "assets"
    / "dexmal_flash_pi0_libero_drafts.json"
)


@pytest.fixture
def registry() -> DraftAssetRegistry:
    return DraftAssetRegistry.from_path(_REGISTRY_PATH)


def test_published_registry_locks_source_contract_and_all_four_assets(
    registry: DraftAssetRegistry,
) -> None:
    assert registry.model_revision == PUBLISHED_MODEL_REVISION
    assert registry.code_revision == PUBLISHED_CODE_REVISION
    assert registry.suites == (
        "libero_10",
        "libero_goal",
        "libero_object",
        "libero_spatial",
    )

    contract = registry.base_contract
    assert contract.model_family == "pi0"
    assert contract.config_name == "pi0_libero"
    assert contract.action_horizon == 50
    assert contract.model_action_dim == 32
    assert contract.environment_action_dim == 7
    assert contract.state_projection_dim == 32
    assert contract.prefix_embedding_dim == 2048
    assert contract.max_token_length == 48
    assert contract.normalization == "z_score"
    assert contract.extra_delta_transform is True

    assert registry.asset("libero_10").sha256 == (
        "371d809745d688e03b2a09ccb89b509906530c7db48117236caf13e676333f17"
    )
    assert registry.asset("libero_goal").sha256 == (
        "33724426753c3b3a4ad5b0c3510bf5ac9255ee6338724bb5c58c7ea8b60429a1"
    )
    assert registry.asset("libero_object").sha256 == (
        "ef4fec6fa4d7d013eb5d54123501159e066d80fa5d23f96690d5bd8aaf397c0a"
    )
    assert registry.asset("libero_spatial").sha256 == (
        "88ef0bbf2ff89d51beaa742fc8a9cb0b1cbfc027282182e1f366f93895fed622"
    )


@pytest.mark.parametrize(
    ("section", "key", "replacement", "message"),
    [
        ("source", "model_revision", "0" * 40, "source identity"),
        ("base_contract", "model_family", "pi05", "base contract"),
    ],
)
def test_registry_rejects_tampered_source_or_contract(
    tmp_path: Path,
    section: str,
    key: str,
    replacement: object,
    message: str,
) -> None:
    data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    data[section][key] = replacement
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        DraftAssetRegistry.from_path(path)


def test_registry_rejects_tampered_published_weight_identity(tmp_path: Path) -> None:
    data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    data["assets"][1]["size"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="asset identity mismatch"):
        DraftAssetRegistry.from_path(path)


def test_matching_pi0_libero_target_is_production_compatible(
    registry: DraftAssetRegistry,
) -> None:
    result = registry.require_production_target(registry.base_contract)

    assert result.purpose == "production"
    assert result.production_compatible is True
    assert result.initialization_allowed is False
    assert result.mismatches == ()


@pytest.mark.parametrize(
    ("changes", "expected_field"),
    [
        ({"model_family": "pi05"}, "model_family"),
        (
            {"base_checkpoint_id": "RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"},
            "base_checkpoint_id",
        ),
        ({"transform_id": "openpi:Pi05LiberoDataConfig"}, "transform_id"),
        ({"normalization": "quantile"}, "normalization"),
        ({"norm_stats_id": "physical-intelligence/robotwin"}, "norm_stats_id"),
        ({"extra_delta_transform": False}, "extra_delta_transform"),
    ],
)
def test_production_gate_rejects_family_base_transform_and_norm_mismatches(
    registry: DraftAssetRegistry,
    changes: dict[str, object],
    expected_field: str,
) -> None:
    target = replace(registry.base_contract, **changes)

    with pytest.raises(ConfigurationError, match=expected_field):
        registry.require_production_target(target)


def test_default_resolution_rejects_pi05_before_touching_unverified_file(
    registry: DraftAssetRegistry, tmp_path: Path
) -> None:
    pi05_target = replace(registry.base_contract, model_family="pi05")

    with pytest.raises(ConfigurationError, match="model_family"):
        registry.resolve("libero_goal", tmp_path / "missing", pi05_target)


def test_explicit_warm_start_is_initialization_only_even_for_exact_base(
    registry: DraftAssetRegistry,
) -> None:
    result = registry.authorize_warm_start(registry.base_contract)

    assert result.purpose == "warm_start_initialization_only"
    assert result.initialization_allowed is True
    assert result.production_compatible is False
    assert result.mismatches == ()


def test_explicit_pi05_warm_start_reports_mismatch_without_certifying_production(
    registry: DraftAssetRegistry,
) -> None:
    pi05_target = replace(
        registry.base_contract,
        model_family="pi05",
        normalization="quantile",
        extra_delta_transform=False,
    )

    result = registry.authorize_warm_start(pi05_target)

    assert result.initialization_allowed is True
    assert result.production_compatible is False
    assert any(value.startswith("model_family:") for value in result.mismatches)
    assert any(value.startswith("normalization:") for value in result.mismatches)
    assert any(
        value.startswith("extra_delta_transform:") for value in result.mismatches
    )


def test_local_asset_verification_checks_size_and_sha256(tmp_path: Path) -> None:
    payload = b"content-addressed Draft test payload\n"
    path = tmp_path / "draft_test.pt"
    path.write_bytes(payload)
    asset = DraftAsset(
        suite="test",
        filename=Path(path.name),
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    verified = asset.verify(tmp_path)

    assert verified.path == path.resolve()
    assert verified.asset is asset


def test_local_asset_verification_rejects_size_then_hash_mismatch(
    tmp_path: Path,
) -> None:
    payload = b"draft"
    path = tmp_path / "draft_test.pt"
    path.write_bytes(payload)

    wrong_size = DraftAsset(
        suite="test",
        filename=Path(path.name),
        size=len(payload) + 1,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ConfigurationError, match="size mismatch"):
        wrong_size.verify(tmp_path)

    wrong_hash = replace(wrong_size, size=len(payload), sha256="0" * 64)
    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        wrong_hash.verify(tmp_path)


def test_target_contract_rejects_incomplete_geometry() -> None:
    with pytest.raises(ConfigurationError, match="action_horizon"):
        DraftTargetContract(
            model_family="pi0",
            config_name="pi0_libero",
            base_checkpoint_id="base",
            action_horizon=0,
            model_action_dim=32,
            environment_action_dim=7,
            state_projection_dim=32,
            prefix_embedding_dim=2048,
            max_token_length=48,
            normalization="z_score",
            norm_stats_id="stats",
            transform_id="transform",
            extra_delta_transform=True,
        )
