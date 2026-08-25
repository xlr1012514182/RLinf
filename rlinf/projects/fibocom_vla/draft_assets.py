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

"""Pinned Realtime-VLA FLASH Draft assets and compatibility gates.

The published Draft heads are trained for the original ``pi0_libero`` base
policy.  They are not interchangeable with a pi0.5 checkpoint merely because
both models emit 50-step chunks.  This module binds each weight file to its
base checkpoint, preprocessing transform, normalization statistics, and
tensor geometry, then fails closed before the file can enter a production
runtime.

An explicit warm-start resolution permits loading the verified bytes only as
initialization for subsequent target-domain training.  It never certifies the
published head as production-compatible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError

PUBLISHED_MODEL_REPOSITORY = "Dexmal/RealtimeVLA-Flash"
PUBLISHED_MODEL_REVISION = "77b9a6f88fb100230bc78cb4cb361bd2e586f9fb"
PUBLISHED_CODE_REPOSITORY = "https://github.com/dexmal/realtime-vla-flash"
PUBLISHED_CODE_REVISION = "da6ceccad603695a8a3d6fa14dd410c3aadb536f"

# This table is deliberately duplicated from the checked-in JSON registry.
# The Python implementation is the trust root, while the JSON file is the
# inspectable/download-facing representation.  Editing JSON alone therefore
# cannot authorize a different checkpoint in production.
_PUBLISHED_ASSETS: Mapping[str, tuple[str, int, str]] = {
    "libero_10": (
        "draft_libero_10.pt",
        441_154_745,
        "371d809745d688e03b2a09ccb89b509906530c7db48117236caf13e676333f17",
    ),
    "libero_goal": (
        "draft_libero_goal.pt",
        441_154_745,
        "33724426753c3b3a4ad5b0c3510bf5ac9255ee6338724bb5c58c7ea8b60429a1",
    ),
    "libero_object": (
        "draft_libero_object.pt",
        441_154_889,
        "ef4fec6fa4d7d013eb5d54123501159e066d80fa5d23f96690d5bd8aaf397c0a",
    ),
    "libero_spatial": (
        "draft_libero_spatial.pt",
        441_154_813,
        "88ef0bbf2ff89d51beaa742fc8a9cb0b1cbfc027282182e1f366f93895fed622",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Draft registry {field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Draft registry {field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"Draft registry {field} must be a positive integer")
    return value


def _sha256_string(value: Any, *, field: str) -> str:
    digest = _string(value, field=field).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ConfigurationError(f"Draft registry {field} is not a SHA-256 digest")
    return digest


def _relative_filename(value: Any, *, field: str) -> Path:
    raw = _string(value, field=field).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() or len(path.parts) != 1 or path.name != raw:
        raise ConfigurationError(
            f"Draft registry {field} must be a single filename inside the asset root"
        )
    return path


@dataclass(frozen=True)
class DraftTargetContract:
    """Semantic and tensor contract of the base policy used by a Draft head."""

    model_family: str
    config_name: str
    base_checkpoint_id: str
    action_horizon: int
    model_action_dim: int
    environment_action_dim: int
    state_projection_dim: int
    prefix_embedding_dim: int
    max_token_length: int
    normalization: str
    norm_stats_id: str
    transform_id: str
    extra_delta_transform: bool

    def __post_init__(self) -> None:
        """Reject incomplete contracts before compatibility is evaluated."""

        string_fields = (
            "model_family",
            "config_name",
            "base_checkpoint_id",
            "normalization",
            "norm_stats_id",
            "transform_id",
        )
        integer_fields = (
            "action_horizon",
            "model_action_dim",
            "environment_action_dim",
            "state_projection_dim",
            "prefix_embedding_dim",
            "max_token_length",
        )
        for name in string_fields:
            _string(getattr(self, name), field=f"target.{name}")
        for name in integer_fields:
            _positive_int(getattr(self, name), field=f"target.{name}")
        if not isinstance(self.extra_delta_transform, bool):
            raise ConfigurationError(
                "Draft registry target.extra_delta_transform must be a boolean"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DraftTargetContract":
        """Parse a complete target/base contract from JSON data."""

        boolean = data.get("extra_delta_transform")
        if not isinstance(boolean, bool):
            raise ConfigurationError(
                "Draft registry base_contract.extra_delta_transform must be a boolean"
            )
        return cls(
            model_family=_string(
                data.get("model_family"), field="base_contract.model_family"
            ),
            config_name=_string(
                data.get("config_name"), field="base_contract.config_name"
            ),
            base_checkpoint_id=_string(
                data.get("base_checkpoint_id"),
                field="base_contract.base_checkpoint_id",
            ),
            action_horizon=_positive_int(
                data.get("action_horizon"), field="base_contract.action_horizon"
            ),
            model_action_dim=_positive_int(
                data.get("model_action_dim"), field="base_contract.model_action_dim"
            ),
            environment_action_dim=_positive_int(
                data.get("environment_action_dim"),
                field="base_contract.environment_action_dim",
            ),
            state_projection_dim=_positive_int(
                data.get("state_projection_dim"),
                field="base_contract.state_projection_dim",
            ),
            prefix_embedding_dim=_positive_int(
                data.get("prefix_embedding_dim"),
                field="base_contract.prefix_embedding_dim",
            ),
            max_token_length=_positive_int(
                data.get("max_token_length"),
                field="base_contract.max_token_length",
            ),
            normalization=_string(
                data.get("normalization"), field="base_contract.normalization"
            ),
            norm_stats_id=_string(
                data.get("norm_stats_id"), field="base_contract.norm_stats_id"
            ),
            transform_id=_string(
                data.get("transform_id"), field="base_contract.transform_id"
            ),
            extra_delta_transform=boolean,
        )

    def mismatches(self, target: "DraftTargetContract") -> tuple[str, ...]:
        """Return every semantic difference from ``target`` in stable order."""

        differences: list[str] = []
        for contract_field in fields(self):
            name = contract_field.name
            published = getattr(self, name)
            requested = getattr(target, name)
            if published != requested:
                differences.append(
                    f"{name}: published={published!r}, target={requested!r}"
                )
        return tuple(differences)


_PUBLISHED_BASE_CONTRACT = DraftTargetContract(
    model_family="pi0",
    config_name="pi0_libero",
    base_checkpoint_id="gs://openpi-assets/checkpoints/pi0_libero",
    action_horizon=50,
    model_action_dim=32,
    environment_action_dim=7,
    state_projection_dim=32,
    prefix_embedding_dim=2048,
    max_token_length=48,
    normalization="z_score",
    norm_stats_id="physical-intelligence/libero",
    transform_id="openpi:LeRobotLiberoDataConfig:delta_first_6",
    extra_delta_transform=True,
)


@dataclass(frozen=True)
class DraftAsset:
    """One published, content-addressed Draft checkpoint file."""

    suite: str
    filename: Path
    size: int
    sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int) -> "DraftAsset":
        """Parse a Draft file entry."""

        prefix = f"assets[{index}]"
        return cls(
            suite=_string(data.get("suite"), field=f"{prefix}.suite"),
            filename=_relative_filename(
                data.get("filename"), field=f"{prefix}.filename"
            ),
            size=_positive_int(data.get("size"), field=f"{prefix}.size"),
            sha256=_sha256_string(data.get("sha256"), field=f"{prefix}.sha256"),
        )

    def verify(self, root: str | Path) -> "VerifiedDraftAsset":
        """Verify this checkpoint's path, byte size, and SHA-256."""

        asset_root = Path(root).expanduser().resolve()
        if not asset_root.is_dir():
            raise ConfigurationError(
                f"Draft checkpoint root is not a directory: {asset_root}"
            )
        candidate = (asset_root / self.filename).resolve()
        if candidate.parent != asset_root:
            raise ConfigurationError(
                f"Draft checkpoint escapes its declared root: {self.filename}"
            )
        if not candidate.is_file():
            raise ConfigurationError(f"Draft checkpoint is missing: {candidate}")
        observed_size = candidate.stat().st_size
        if observed_size != self.size:
            raise ConfigurationError(
                f"Draft checkpoint size mismatch for {self.filename}: "
                f"expected {self.size}, got {observed_size}"
            )
        observed_sha256 = _sha256(candidate)
        if observed_sha256 != self.sha256:
            raise ConfigurationError(
                f"Draft checkpoint SHA-256 mismatch for {self.filename}: "
                f"expected {self.sha256}, got {observed_sha256}"
            )
        return VerifiedDraftAsset(asset=self, path=candidate)


@dataclass(frozen=True)
class VerifiedDraftAsset:
    """Local Draft checkpoint whose bytes match the source-locked registry."""

    asset: DraftAsset
    path: Path


@dataclass(frozen=True)
class DraftCompatibility:
    """Authorization result for one narrowly defined use of a Draft asset."""

    purpose: str
    production_compatible: bool
    initialization_allowed: bool
    mismatches: tuple[str, ...]


@dataclass(frozen=True)
class DraftResolution:
    """Verified bytes paired with a production or warm-start authorization."""

    verified_asset: VerifiedDraftAsset
    compatibility: DraftCompatibility

    @property
    def path(self) -> Path:
        """Return the verified local checkpoint path."""

        return self.verified_asset.path

    @property
    def production_compatible(self) -> bool:
        """Return whether direct production use was authorized."""

        return self.compatibility.production_compatible


@dataclass(frozen=True)
class DraftAssetRegistry:
    """Source-locked set of published Realtime-VLA FLASH Draft heads."""

    model_repository: str
    model_revision: str
    code_repository: str
    code_revision: str
    base_contract: DraftTargetContract
    assets: tuple[DraftAsset, ...]
    manifest_path: Path

    @classmethod
    def from_path(cls, path: str | Path) -> "DraftAssetRegistry":
        """Load and authenticate the checked-in published-asset registry."""

        manifest_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"failed to read Draft asset registry {manifest_path}: {error}"
            ) from error
        data = _object(raw, field="root")
        if data.get("schema_version") != 1:
            raise ConfigurationError("unsupported Draft asset registry schema")

        source = _object(data.get("source"), field="source")
        registry = cls(
            model_repository=_string(
                source.get("model_repository"), field="source.model_repository"
            ),
            model_revision=_string(
                source.get("model_revision"), field="source.model_revision"
            ),
            code_repository=_string(
                source.get("code_repository"), field="source.code_repository"
            ),
            code_revision=_string(
                source.get("code_revision"), field="source.code_revision"
            ),
            base_contract=DraftTargetContract.from_dict(
                _object(data.get("base_contract"), field="base_contract")
            ),
            assets=tuple(
                DraftAsset.from_dict(
                    _object(value, field=f"assets[{index}]"), index=index
                )
                for index, value in enumerate(cls._asset_values(data))
            ),
            manifest_path=manifest_path,
        )
        registry._validate_published_identity()
        return registry

    @staticmethod
    def _asset_values(data: Mapping[str, Any]) -> list[Any]:
        values = data.get("assets")
        if not isinstance(values, list) or not values:
            raise ConfigurationError("Draft registry assets must be a non-empty list")
        return values

    def _validate_published_identity(self) -> None:
        expected_source = (
            PUBLISHED_MODEL_REPOSITORY,
            PUBLISHED_MODEL_REVISION,
            PUBLISHED_CODE_REPOSITORY,
            PUBLISHED_CODE_REVISION,
        )
        observed_source = (
            self.model_repository,
            self.model_revision,
            self.code_repository,
            self.code_revision,
        )
        if observed_source != expected_source:
            raise ConfigurationError(
                "Draft registry source identity is not the audited published revision"
            )
        contract_mismatches = _PUBLISHED_BASE_CONTRACT.mismatches(self.base_contract)
        if contract_mismatches:
            raise ConfigurationError(
                "Draft registry base contract differs from the audited publication: "
                + "; ".join(contract_mismatches)
            )

        by_suite = {asset.suite: asset for asset in self.assets}
        if len(by_suite) != len(self.assets):
            raise ConfigurationError("Draft registry suite names must be unique")
        if set(by_suite) != set(_PUBLISHED_ASSETS):
            raise ConfigurationError(
                "Draft registry must contain exactly the four audited LIBERO heads"
            )
        for suite, expected in _PUBLISHED_ASSETS.items():
            asset = by_suite[suite]
            observed = (str(asset.filename), asset.size, asset.sha256)
            if observed != expected:
                raise ConfigurationError(
                    f"Draft registry asset identity mismatch for {suite}: "
                    f"expected {expected!r}, got {observed!r}"
                )

    @property
    def suites(self) -> tuple[str, ...]:
        """Return the available suites in registry order."""

        return tuple(asset.suite for asset in self.assets)

    def asset(self, suite: str) -> DraftAsset:
        """Return one suite's pinned checkpoint declaration."""

        for asset in self.assets:
            if asset.suite == suite:
                return asset
        raise ConfigurationError(
            f"unknown Draft suite {suite!r}; expected one of {', '.join(self.suites)}"
        )

    def verify_asset(self, suite: str, root: str | Path) -> VerifiedDraftAsset:
        """Resolve and cryptographically verify one local Draft checkpoint."""

        return self.asset(suite).verify(root)

    def require_production_target(
        self, target: DraftTargetContract
    ) -> DraftCompatibility:
        """Require exact source-base compatibility for direct inference."""

        mismatches = self.base_contract.mismatches(target)
        if mismatches:
            raise ConfigurationError(
                "published Draft checkpoint is not production-compatible with the "
                "target policy: " + "; ".join(mismatches)
            )
        return DraftCompatibility(
            purpose="production",
            production_compatible=True,
            initialization_allowed=False,
            mismatches=(),
        )

    def authorize_warm_start(self, target: DraftTargetContract) -> DraftCompatibility:
        """Authorize verified weights only as non-production initialization.

        This method intentionally never returns ``production_compatible=True``.
        A caller must train and independently validate a target-domain Draft head
        before constructing a new production registry for that derived asset.
        """

        return DraftCompatibility(
            purpose="warm_start_initialization_only",
            production_compatible=False,
            initialization_allowed=True,
            mismatches=self.base_contract.mismatches(target),
        )

    def resolve(
        self,
        suite: str,
        root: str | Path,
        target: DraftTargetContract,
    ) -> DraftResolution:
        """Verify and resolve a Draft checkpoint for production (fail closed)."""

        compatibility = self.require_production_target(target)
        verified = self.verify_asset(suite, root)
        return DraftResolution(
            verified_asset=verified,
            compatibility=compatibility,
        )

    def resolve_warm_start(
        self,
        suite: str,
        root: str | Path,
        target: DraftTargetContract,
    ) -> DraftResolution:
        """Verify and resolve weights for initialization-only warm-start use."""

        compatibility = self.authorize_warm_start(target)
        verified = self.verify_asset(suite, root)
        return DraftResolution(
            verified_asset=verified,
            compatibility=compatibility,
        )


def load_draft_asset_registry(path: str | Path) -> DraftAssetRegistry:
    """Load the fixed published Draft registry from ``path``."""

    return DraftAssetRegistry.from_path(path)


__all__ = [
    "DraftAsset",
    "DraftAssetRegistry",
    "DraftCompatibility",
    "DraftResolution",
    "DraftTargetContract",
    "PUBLISHED_CODE_REPOSITORY",
    "PUBLISHED_CODE_REVISION",
    "PUBLISHED_MODEL_REPOSITORY",
    "PUBLISHED_MODEL_REVISION",
    "VerifiedDraftAsset",
    "load_draft_asset_registry",
]
