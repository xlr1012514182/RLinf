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

"""Source-locked checkpoint manifests and compatibility gates.

The model weights, normalization statistics, transforms, and camera contract
form one inseparable policy asset.  A matching tensor shape is insufficient:
connecting a checkpoint to the wrong transform can produce finite but unsafe
robot commands.  This module therefore verifies the complete declared asset
before any heavyweight OpenPI model is instantiated.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import StackConfig
from .errors import ConfigurationError

# An identity-only capability: successful byte/source verification is the only
# code path in this module that attaches this object to a result.  The proof is
# deliberately not serialized and is excluded from dataclass equality/repr.
_VERIFIED_CHECKPOINT_CAPABILITY = object()


def _sha256(path: Path, *, normalize_newlines: bool = False) -> str:
    digest = hashlib.sha256()
    if normalize_newlines:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"asset manifest {field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"asset manifest {field} must be a non-empty string")
    return value.strip()


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"asset manifest {field} must be a positive integer")
    return value


def _string_tuple(value: Any, *, field: str, length: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ConfigurationError(
            f"asset manifest {field} must be a list with {length} entries"
        )
    result = tuple(
        _string(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ConfigurationError(f"asset manifest {field} entries must be unique")
    return result


def _boolean_tuple(value: Any, *, field: str, length: int) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, bool) for item in value)
    ):
        raise ConfigurationError(
            f"asset manifest {field} must contain {length} explicit booleans"
        )
    return tuple(value)


def _relative_path(value: Any, *, field: str) -> Path:
    raw = _string(value, field=field).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(
            f"asset manifest {field} must stay inside the declared root"
        )
    return path


@dataclass(frozen=True)
class AssetFile:
    """One required file in a checkpoint directory."""

    path: Path
    size: int
    sha256: str
    role: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int) -> "AssetFile":
        prefix = f"files[{index}]"
        sha256 = _string(data.get("sha256"), field=f"{prefix}.sha256").lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ConfigurationError(f"asset manifest {prefix}.sha256 is invalid")
        return cls(
            path=_relative_path(data.get("path"), field=f"{prefix}.path"),
            size=_integer(data.get("size"), field=f"{prefix}.size"),
            sha256=sha256,
            role=_string(data.get("role"), field=f"{prefix}.role"),
        )


@dataclass(frozen=True)
class SourceFingerprint:
    """Canonical-LF SHA-256 for runtime source that defines policy semantics."""

    path: Path
    sha256_lf: str
    repository: str
    revision: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int) -> "SourceFingerprint":
        prefix = f"runtime.source_fingerprints[{index}]"
        digest = _string(data.get("sha256_lf"), field=f"{prefix}.sha256_lf").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ConfigurationError(f"asset manifest {prefix}.sha256_lf is invalid")
        return cls(
            path=_relative_path(data.get("path"), field=f"{prefix}.path"),
            sha256_lf=digest,
            repository=_string(data.get("repository"), field=f"{prefix}.repository"),
            revision=_string(data.get("revision"), field=f"{prefix}.revision"),
        )


@dataclass(frozen=True)
class CameraContract:
    """Ordered observation keys consumed by an OpenPI data transform."""

    main: str
    wrists: tuple[str, ...]
    model_image_keys: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraContract":
        wrists = data.get("wrists", [])
        model_keys = data.get("model_image_keys", [])
        if not isinstance(wrists, list) or not isinstance(model_keys, list):
            raise ConfigurationError(
                "asset manifest runtime.cameras wrists/model_image_keys must be lists"
            )
        contract = cls(
            main=_string(data.get("main"), field="runtime.cameras.main"),
            wrists=tuple(
                _string(value, field=f"runtime.cameras.wrists[{index}]")
                for index, value in enumerate(wrists)
            ),
            model_image_keys=tuple(
                _string(value, field=f"runtime.cameras.model_image_keys[{index}]")
                for index, value in enumerate(model_keys)
            ),
        )
        all_inputs = (contract.main, *contract.wrists)
        if len(set(all_inputs)) != len(all_inputs):
            raise ConfigurationError("asset manifest camera input keys must be unique")
        if len(contract.model_image_keys) != len(all_inputs):
            raise ConfigurationError(
                "asset manifest model image keys must match the input camera count"
            )
        return contract

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return all required observation camera keys in transform order."""

        return (self.main, *self.wrists)


@dataclass(frozen=True)
class RuntimeContract:
    """OpenPI geometry and preprocessing identity bound to a checkpoint."""

    model_family: str
    config_name: str
    asset_id: str
    action_horizon: int
    model_action_dim: int
    max_token_length: int
    discrete_state_input: bool
    environment_action_dim: int
    state_dim: int
    model_state_dim: int
    state_semantics: tuple[str, ...]
    environment_action_semantics: tuple[str, ...]
    delta_action_mask: tuple[bool, ...]
    normalization: str
    norm_stats_path: Path
    checkpoint_config_path: Path
    action_mode: str
    policy_frame: str
    extra_delta_transform: bool
    adapt_to_pi: bool
    cameras: CameraContract
    source_fingerprints: tuple[SourceFingerprint, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeContract":
        source_values = data.get("source_fingerprints", [])
        if not isinstance(source_values, list) or not source_values:
            raise ConfigurationError(
                "asset manifest runtime.source_fingerprints must be a non-empty list"
            )
        booleans = (
            "discrete_state_input",
            "extra_delta_transform",
            "adapt_to_pi",
        )
        if any(not isinstance(data.get(name), bool) for name in booleans):
            raise ConfigurationError(
                "asset manifest transform flags must be explicit booleans"
            )
        environment_action_dim = _integer(
            data.get("environment_action_dim"),
            field="runtime.environment_action_dim",
        )
        state_dim = _integer(data.get("state_dim"), field="runtime.state_dim")
        model_state_dim = _integer(
            data.get("model_state_dim"), field="runtime.model_state_dim"
        )
        delta_action_mask = _boolean_tuple(
            data.get("delta_action_mask"),
            field="runtime.delta_action_mask",
            length=environment_action_dim,
        )
        extra_delta_transform = data["extra_delta_transform"]
        if extra_delta_transform and not any(delta_action_mask):
            raise ConfigurationError(
                "asset manifest delta_action_mask must select at least one "
                "coordinate when extra_delta_transform=true"
            )
        if not extra_delta_transform and any(delta_action_mask):
            raise ConfigurationError(
                "asset manifest delta_action_mask must be all false when "
                "extra_delta_transform=false"
            )
        runtime = cls(
            model_family=_string(
                data.get("model_family"), field="runtime.model_family"
            ),
            config_name=_string(data.get("config_name"), field="runtime.config_name"),
            asset_id=_string(data.get("asset_id"), field="runtime.asset_id"),
            action_horizon=_integer(
                data.get("action_horizon"), field="runtime.action_horizon"
            ),
            model_action_dim=_integer(
                data.get("model_action_dim"), field="runtime.model_action_dim"
            ),
            max_token_length=_integer(
                data.get("max_token_length"), field="runtime.max_token_length"
            ),
            discrete_state_input=data["discrete_state_input"],
            environment_action_dim=environment_action_dim,
            state_dim=state_dim,
            model_state_dim=model_state_dim,
            state_semantics=_string_tuple(
                data.get("state_semantics"),
                field="runtime.state_semantics",
                length=state_dim,
            ),
            environment_action_semantics=_string_tuple(
                data.get("environment_action_semantics"),
                field="runtime.environment_action_semantics",
                length=environment_action_dim,
            ),
            delta_action_mask=delta_action_mask,
            normalization=_string(
                data.get("normalization"), field="runtime.normalization"
            ),
            norm_stats_path=_relative_path(
                data.get("norm_stats_path"), field="runtime.norm_stats_path"
            ),
            checkpoint_config_path=_relative_path(
                data.get("checkpoint_config_path"),
                field="runtime.checkpoint_config_path",
            ),
            action_mode=_string(data.get("action_mode"), field="runtime.action_mode"),
            policy_frame=_string(
                data.get("policy_frame"), field="runtime.policy_frame"
            ),
            extra_delta_transform=extra_delta_transform,
            adapt_to_pi=data["adapt_to_pi"],
            cameras=CameraContract.from_dict(
                _mapping(data.get("cameras"), field="runtime.cameras")
            ),
            source_fingerprints=tuple(
                SourceFingerprint.from_dict(
                    _mapping(value, field=f"runtime.source_fingerprints[{index}]"),
                    index=index,
                )
                for index, value in enumerate(source_values)
            ),
        )
        if runtime.normalization != "quantile_q01_q99":
            raise ConfigurationError(
                "asset manifest runtime.normalization must be "
                "'quantile_q01_q99' for a source-locked pi0.5 asset"
            )
        if runtime.model_state_dim != 32:
            raise ConfigurationError(
                "asset manifest runtime.model_state_dim must be 32 for RLinf pi0.5"
            )
        if runtime.state_dim > runtime.model_state_dim:
            raise ConfigurationError(
                "asset manifest raw state_dim exceeds model_state_dim"
            )
        return runtime


@dataclass(frozen=True)
class VerifiedCheckpointAssets:
    """Fully verified local resolution of a checkpoint manifest."""

    manifest: "CheckpointAssetManifest"
    root: Path
    manifest_path: Path
    _verification_proof: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def checkpoint_id(self) -> str:
        """Return a revision-bound checkpoint identity for downstream gates."""

        return f"{self.manifest.repository_id}@{self.manifest.revision}"

    @property
    def norm_stats_path(self) -> Path:
        """Return the verified normalization-statistics file."""

        return self.root / self.manifest.runtime.norm_stats_path

    @property
    def weight_paths(self) -> tuple[Path, ...]:
        """Return the exact verified safetensors shards consumed by RLinf."""

        return tuple(
            self.root / file.path
            for file in self.manifest.files
            if file.role == "weights_shard"
        )


@dataclass(frozen=True)
class CheckpointAssetManifest:
    """Immutable checkpoint identity and executable preprocessing contract."""

    schema_version: int
    repository_id: str
    repository_url: str
    revision: str
    runtime: RuntimeContract
    publisher_config: Mapping[str, Any]
    files: tuple[AssetFile, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> "CheckpointAssetManifest":
        """Parse and validate a JSON asset manifest."""

        manifest_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"failed to read checkpoint asset manifest {manifest_path}: {error}"
            ) from error
        data = _mapping(raw, field="root")
        if data.get("schema_version") != 1:
            raise ConfigurationError("unsupported checkpoint asset manifest schema")
        identity = _mapping(data.get("identity"), field="identity")
        file_values = data.get("files")
        if not isinstance(file_values, list) or not file_values:
            raise ConfigurationError("asset manifest files must be a non-empty list")
        files = tuple(
            AssetFile.from_dict(_mapping(value, field=f"files[{index}]"), index=index)
            for index, value in enumerate(file_values)
        )
        paths = tuple(file.path for file in files)
        if len(set(paths)) != len(paths):
            raise ConfigurationError("asset manifest file paths must be unique")
        runtime = RuntimeContract.from_dict(
            _mapping(data.get("runtime"), field="runtime")
        )
        by_path = {file.path: file for file in files}
        required_runtime_files = {
            runtime.norm_stats_path: "selected_norm_stats",
            runtime.checkpoint_config_path: "publisher_config",
        }
        for required_path, required_role in required_runtime_files.items():
            declared = by_path.get(required_path)
            if declared is None or declared.role != required_role:
                raise ConfigurationError(
                    f"asset manifest {required_path} must be declared with role "
                    f"{required_role!r}"
                )
        weight_indexes = tuple(file for file in files if file.role == "weights_index")
        weight_shards = tuple(file for file in files if file.role == "weights_shard")
        if len(weight_indexes) != 1 or not weight_shards:
            raise ConfigurationError(
                "asset manifest must declare exactly one weights_index and at "
                "least one weights_shard"
            )
        if (
            weight_indexes[0].path.parent != Path()
            or weight_indexes[0].path.suffix != ".json"
        ):
            raise ConfigurationError(
                "weights_index must be a checkpoint-root JSON file"
            )
        if any(
            file.path.parent != Path() or file.path.suffix != ".safetensors"
            for file in weight_shards
        ):
            raise ConfigurationError(
                "every weights_shard must be a checkpoint-root .safetensors file"
            )
        return cls(
            schema_version=1,
            repository_id=_string(
                identity.get("repository_id"), field="identity.repository_id"
            ),
            repository_url=_string(
                identity.get("repository_url"), field="identity.repository_url"
            ),
            revision=_string(identity.get("revision"), field="identity.revision"),
            runtime=runtime,
            publisher_config=dict(
                _mapping(data.get("publisher_config"), field="publisher_config")
            ),
            files=files,
        )

    def verify(
        self,
        root: str | Path,
        *,
        manifest_path: str | Path,
        repository_root: str | Path | None = None,
        verify_hashes: bool = True,
    ) -> VerifiedCheckpointAssets:
        """Verify all required bytes, stats geometry, and runtime sources."""

        if not verify_hashes:
            raise ConfigurationError(
                "SHA-256 verification cannot be disabled for verified checkpoint assets"
            )
        asset_root = Path(root).expanduser().resolve()
        if not asset_root.is_dir():
            raise ConfigurationError(
                f"checkpoint root is not a directory: {asset_root}"
            )
        if repository_root is None:
            raise ConfigurationError(
                "repository_root is required to verify runtime source fingerprints"
            )
        repository_path = Path(repository_root).expanduser().resolve()
        self._verify_source_fingerprints(repository_path)
        self._verify_loader_resolution(asset_root)
        for file in self.files:
            candidate = (asset_root / file.path).resolve()
            if asset_root not in candidate.parents:
                raise ConfigurationError(
                    f"checkpoint path escapes asset root: {file.path}"
                )
            if not candidate.is_file():
                raise ConfigurationError(f"checkpoint asset is missing: {candidate}")
            observed_size = candidate.stat().st_size
            if observed_size != file.size:
                raise ConfigurationError(
                    f"checkpoint asset size mismatch for {file.path}: "
                    f"expected {file.size}, got {observed_size}"
                )
            if verify_hashes:
                observed_sha256 = _sha256(candidate)
                if observed_sha256 != file.sha256:
                    raise ConfigurationError(
                        f"checkpoint asset SHA-256 mismatch for {file.path}: "
                        f"expected {file.sha256}, got {observed_sha256}"
                    )

        self._verify_publisher_config(asset_root)
        self._verify_norm_stats(asset_root)
        self._verify_weight_index(asset_root)
        return VerifiedCheckpointAssets(
            manifest=self,
            root=asset_root,
            manifest_path=Path(manifest_path).expanduser().resolve(),
            _verification_proof=_VERIFIED_CHECKPOINT_CAPABILITY,
        )

    def _verify_loader_resolution(self, root: Path) -> None:
        """Ensure RLinf cannot select bytes outside the verified shard set."""

        shadow_paths = (
            root / "model_state_dict" / "full_weights.pt",
            root / "actor" / "model_state_dict" / "full_weights.pt",
        )
        present_shadows = tuple(path for path in shadow_paths if path.exists())
        if present_shadows:
            raise ConfigurationError(
                "unverified full_weights.pt would shadow manifest shards: "
                + ", ".join(str(path) for path in present_shadows)
            )
        expected = {
            (root / file.path).resolve()
            for file in self.files
            if file.role == "weights_shard"
        }
        observed = {path.resolve() for path in root.glob("*.safetensors")}
        if observed != expected:
            missing = sorted(str(path) for path in expected - observed)
            extra = sorted(str(path) for path in observed - expected)
            raise ConfigurationError(
                "checkpoint-root safetensors set differs from the manifest: "
                f"missing={missing}, extra={extra}"
            )

    def _verify_weight_index(self, root: Path) -> None:
        """Bind the safetensors index to exactly the declared shard set."""

        index_file = next(file for file in self.files if file.role == "weights_index")
        path = root / index_file.path
        try:
            data = _mapping(
                json.loads(path.read_text(encoding="utf-8")),
                field="weights_index",
            )
            weight_map = _mapping(
                data.get("weight_map"), field="weights_index.weight_map"
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"invalid safetensors weight index {path}: {error}"
            ) from error
        if not weight_map or any(
            not isinstance(key, str) or not key for key in weight_map
        ):
            raise ConfigurationError(
                "weights_index.weight_map must contain tensor names"
            )
        indexed = {
            _relative_path(value, field=f"weights_index.weight_map[{key!r}]")
            for key, value in weight_map.items()
        }
        if any(path.parent != Path() for path in indexed):
            raise ConfigurationError(
                "weights_index must reference checkpoint-root shard filenames"
            )
        declared = {file.path for file in self.files if file.role == "weights_shard"}
        if indexed != declared:
            raise ConfigurationError(
                "safetensors weight index shard set differs from the manifest: "
                f"indexed={sorted(map(str, indexed))}, "
                f"declared={sorted(map(str, declared))}"
            )

    def _verify_publisher_config(self, root: Path) -> None:
        path = root / self.runtime.checkpoint_config_path
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"invalid publisher checkpoint config {path}: {error}"
            ) from error
        for key, expected in self.publisher_config.items():
            if observed.get(key) != expected:
                raise ConfigurationError(
                    f"publisher checkpoint config mismatch for {key}: "
                    f"expected {expected!r}, got {observed.get(key)!r}"
                )

    def _verify_norm_stats(self, root: Path) -> None:
        path = root / self.runtime.norm_stats_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stats = _mapping(data, field="norm_stats")["norm_stats"]
            stats = _mapping(stats, field="norm_stats.norm_stats")
        except (OSError, json.JSONDecodeError, KeyError) as error:
            raise ConfigurationError(
                f"invalid normalization statistics {path}: {error}"
            ) from error
        expected_dims = {
            "state": self.runtime.state_dim,
            "actions": self.runtime.environment_action_dim,
        }
        required_fields = ("mean", "std", "q01", "q99")
        for name, expected_dim in expected_dims.items():
            group = _mapping(stats.get(name), field=f"norm_stats.{name}")
            for statistic in required_fields:
                values = group.get(statistic)
                if not isinstance(values, list) or len(values) != expected_dim:
                    raise ConfigurationError(
                        f"norm_stats.{name}.{statistic} must have {expected_dim} entries"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in values
                ):
                    raise ConfigurationError(
                        f"norm_stats.{name}.{statistic} contains a non-finite value"
                    )
            if any(value <= 0 for value in group["std"]):
                raise ConfigurationError(f"norm_stats.{name}.std must be positive")
            if any(low > high for low, high in zip(group["q01"], group["q99"])):
                raise ConfigurationError(f"norm_stats.{name} has q01 above q99")

    def _verify_source_fingerprints(self, repository_root: Path) -> None:
        for source in self.runtime.source_fingerprints:
            path = (repository_root / source.path).resolve()
            if repository_root not in path.parents or not path.is_file():
                raise ConfigurationError(f"runtime transform source is missing: {path}")
            observed = _sha256(path, normalize_newlines=True)
            if observed != source.sha256_lf:
                raise ConfigurationError(
                    f"runtime transform SHA-256 mismatch for {source.path}: "
                    f"expected {source.sha256_lf}, got {observed}"
                )

    def validate_stack(self, config: StackConfig) -> None:
        """Reject a model/robot assembly with incompatible policy semantics."""

        runtime = self.runtime
        residual = config.residual_rl
        adapter = config.robot.action_adapter
        mismatches: list[str] = []
        if runtime.model_family != "pi05":
            mismatches.append(f"model_family={runtime.model_family!r}, expected 'pi05'")
        if residual.action_horizon != runtime.action_horizon:
            mismatches.append(
                f"action_horizon stack={residual.action_horizon} asset={runtime.action_horizon}"
            )
        if residual.action_dim != runtime.environment_action_dim:
            mismatches.append(
                "environment action_dim "
                f"stack={residual.action_dim} asset={runtime.environment_action_dim}"
            )
        policy_state_dim = adapter.resolved_policy_state_dim(config.robot.action_dim)
        if policy_state_dim != runtime.state_dim:
            mismatches.append(
                f"policy state_dim stack={policy_state_dim} asset={runtime.state_dim}"
            )
        policy_dim = adapter.resolved_policy_dim(config.robot.action_dim)
        mapping = adapter.robot_from_policy or tuple(range(config.robot.action_dim))
        derived_policy_names = [""] * policy_dim
        if len(mapping) == config.robot.action_dim and all(
            0 <= index < policy_dim for index in mapping
        ):
            for robot_index, policy_index in enumerate(mapping):
                derived_policy_names[policy_index] = config.robot.joint_names[
                    robot_index
                ]
        policy_names = adapter.policy_names or tuple(derived_policy_names)
        policy_state_names = adapter.policy_state_names or policy_names
        coordinate_modes = (
            adapter.coordinate_modes or (adapter.action_mode,) * policy_dim
        )
        expected_coordinate_modes = (
            runtime.action_mode,
        ) * runtime.environment_action_dim
        if tuple(policy_state_names) != runtime.state_semantics:
            mismatches.append(
                "policy_state_names do not match checkpoint state semantics"
            )
        if tuple(policy_names) != runtime.environment_action_semantics:
            mismatches.append(
                "policy_names do not match checkpoint environment-action semantics"
            )
        if tuple(coordinate_modes) != expected_coordinate_modes:
            mismatches.append(
                "coordinate_modes do not match checkpoint environment output mode"
            )
        if adapter.action_mode != runtime.action_mode:
            mismatches.append(
                f"action_mode stack={adapter.action_mode!r} asset={runtime.action_mode!r}"
            )
        if adapter.policy_frame != runtime.policy_frame:
            mismatches.append(
                f"policy_frame stack={adapter.policy_frame!r} asset={runtime.policy_frame!r}"
            )
        camera_names = {camera.name for camera in config.cameras}
        missing_cameras = sorted(set(runtime.cameras.observation_keys) - camera_names)
        if missing_cameras:
            mismatches.append("missing checkpoint cameras=" + ",".join(missing_cameras))
        if mismatches:
            raise ConfigurationError(
                "checkpoint manifest is incompatible with the selected stack: "
                + "; ".join(mismatches)
            )


def load_and_verify_checkpoint_assets(
    manifest_path: str | Path,
    root: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> VerifiedCheckpointAssets:
    """Load a manifest and return its verified local checkpoint resolution."""

    manifest = CheckpointAssetManifest.from_path(manifest_path)
    return manifest.verify(
        root,
        manifest_path=manifest_path,
        repository_root=repository_root,
        verify_hashes=verify_hashes,
    )


def require_verified_checkpoint_assets(value: Any) -> VerifiedCheckpointAssets:
    """Require a result issued by :meth:`CheckpointAssetManifest.verify`.

    Type/field lookalikes and manually instantiated dataclasses have no module
    capability and are rejected.  Consumers must still retain the immutable
    returned object instead of accepting a caller-reported checkpoint string.
    """

    if (
        not isinstance(value, VerifiedCheckpointAssets)
        or value._verification_proof is not _VERIFIED_CHECKPOINT_CAPABILITY
    ):
        raise ConfigurationError(
            "checkpoint assets lack a successful verification proof; use "
            "load_and_verify_checkpoint_assets()"
        )
    return value


__all__ = [
    "AssetFile",
    "CameraContract",
    "CheckpointAssetManifest",
    "RuntimeContract",
    "SourceFingerprint",
    "VerifiedCheckpointAssets",
    "load_and_verify_checkpoint_assets",
    "require_verified_checkpoint_assets",
]
