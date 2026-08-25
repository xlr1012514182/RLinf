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

"""Fail-closed TensorRT build manifests and a TensorRT 10 runtime boundary.

This module deliberately does not export a pi0.5 model.  A caller must provide
either an already-exported ONNX artifact or a callback which populates a
TensorRT network.  Every engine is bound to checkpoint, normalization,
transform, build-input, optimization-profile, and hardware identities before
it can be loaded.

TensorRT and PyTorch are optional and are imported only by build/runtime entry
points.  Manifest parsing and verification remain usable in CPU-only tests.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any
from weakref import WeakKeyDictionary

from ..errors import ConfigurationError, OptionalDependencyError

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_COMPUTE_CAPABILITY_PATTERN = re.compile(r"[0-9]+\.[0-9]+")
_LEGACY_SCHEMA_VERSION = 1
_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {_LEGACY_SCHEMA_VERSION, _SCHEMA_VERSION}


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value


def _strict_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{field} must be a JSON object")
    return value


def _strict_sequence(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{field} must be a JSON array")
    return value


def _require_keys(data: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(data)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ConfigurationError(f"{field} has invalid keys ({'; '.join(details)})")


def _reject_constant(value: str) -> None:
    raise ConfigurationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_strict_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"invalid engine manifest JSON: {error}") from error
    return _strict_mapping(value, field="manifest")


def _canonical_relative_path(value: Any, *, field: str) -> str:
    path = _nonempty_string(value, field=field)
    if "\\" in path:
        raise ConfigurationError(f"{field} must use canonical '/' separators")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ConfigurationError(f"{field} must be a canonical relative path")
    if pure.drive or ":" in pure.parts[0]:
        raise ConfigurationError(f"{field} must not contain a drive prefix")
    canonical = pure.as_posix()
    if canonical != path:
        raise ConfigurationError(f"{field} must be a canonical relative path")
    return canonical


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ConfigurationError(
            f"artifact path escapes verification root: {relative_path}"
        ) from error
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactDigest:
    """Relative artifact locator, exact byte size, and SHA-256 digest."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path", _canonical_relative_path(self.path, field="artifact.path")
        )
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            raise ConfigurationError(
                "artifact.sha256 must be a lowercase 64-character SHA-256"
            )
        _strict_integer(self.size_bytes, field="artifact.size_bytes")

    @classmethod
    def from_path(cls, root: str | Path, path: str) -> ArtifactDigest:
        """Hash one artifact below ``root`` and return its immutable identity."""

        artifact_root = Path(root).expanduser().resolve()
        relative_path = _canonical_relative_path(path, field="artifact.path")
        candidate = _resolve_artifact(artifact_root, relative_path)
        if not candidate.is_file():
            raise ConfigurationError(f"artifact is missing: {candidate}")
        return cls(
            path=relative_path,
            sha256=_sha256(candidate),
            size_bytes=candidate.stat().st_size,
        )

    @classmethod
    def from_dict(cls, value: Any, *, field: str) -> ArtifactDigest:
        """Parse an artifact identity from a strict JSON object."""

        data = _strict_mapping(value, field=field)
        _require_keys(data, {"path", "sha256", "size_bytes"}, field=field)
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            size_bytes=_strict_integer(data["size_bytes"], field=f"{field}.size_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    def verify(self, root: str | Path) -> Path:
        """Verify path containment, byte size, and SHA-256, then return the path."""

        artifact_root = Path(root).expanduser().resolve()
        if not artifact_root.is_dir():
            raise ConfigurationError(
                f"artifact root is not a directory: {artifact_root}"
            )
        candidate = _resolve_artifact(artifact_root, self.path)
        if not candidate.is_file():
            raise ConfigurationError(f"artifact is missing: {candidate}")
        observed_size = candidate.stat().st_size
        if observed_size != self.size_bytes:
            raise ConfigurationError(
                f"artifact size mismatch for {self.path}: "
                f"expected {self.size_bytes}, got {observed_size}"
            )
        observed_digest = _sha256(candidate)
        if observed_digest != self.sha256:
            raise ConfigurationError(
                f"artifact SHA-256 mismatch for {self.path}: "
                f"expected {self.sha256}, got {observed_digest}"
            )
        return candidate


@dataclass(frozen=True)
class SourceCheckpointIdentity:
    """Immutable source checkpoint and preprocessing lineage."""

    revision: str
    files: tuple[ArtifactDigest, ...]
    norm_stats: ArtifactDigest
    transforms: tuple[ArtifactDigest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not _REVISION_PATTERN.fullmatch(
            self.revision
        ):
            raise ConfigurationError(
                "source_checkpoint.revision must be a lowercase 40-64 character "
                "content revision"
            )
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if not self.files:
            raise ConfigurationError("source_checkpoint.files must not be empty")
        if not self.transforms:
            raise ConfigurationError("source_checkpoint.transforms must not be empty")
        artifacts = (*self.files, self.norm_stats, *self.transforms)
        if not all(isinstance(item, ArtifactDigest) for item in artifacts):
            raise TypeError("checkpoint identities must contain ArtifactDigest objects")
        paths = [artifact.path for artifact in artifacts]
        if len(paths) != len(set(paths)):
            raise ConfigurationError("source checkpoint artifact paths must be unique")

    @classmethod
    def from_dict(cls, value: Any) -> SourceCheckpointIdentity:
        """Parse a source checkpoint identity from strict JSON."""

        data = _strict_mapping(value, field="source_checkpoint")
        _require_keys(
            data,
            {"revision", "files", "norm_stats", "transforms"},
            field="source_checkpoint",
        )
        files = _strict_sequence(data["files"], field="source_checkpoint.files")
        transforms = _strict_sequence(
            data["transforms"], field="source_checkpoint.transforms"
        )
        return cls(
            revision=_nonempty_string(
                data["revision"], field="source_checkpoint.revision"
            ),
            files=tuple(
                ArtifactDigest.from_dict(
                    item, field=f"source_checkpoint.files[{index}]"
                )
                for index, item in enumerate(files)
            ),
            norm_stats=ArtifactDigest.from_dict(
                data["norm_stats"], field="source_checkpoint.norm_stats"
            ),
            transforms=tuple(
                ArtifactDigest.from_dict(
                    item, field=f"source_checkpoint.transforms[{index}]"
                )
                for index, item in enumerate(transforms)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "revision": self.revision,
            "files": [item.to_dict() for item in self.files],
            "norm_stats": self.norm_stats.to_dict(),
            "transforms": [item.to_dict() for item in self.transforms],
        }

    def verify(self, root: str | Path) -> None:
        """Verify every checkpoint, normalization, and transform artifact."""

        for artifact in (*self.files, self.norm_stats, *self.transforms):
            artifact.verify(root)


@dataclass(frozen=True)
class InputShapeRange:
    """One TensorRT input's inclusive minimum/optimum/maximum shapes."""

    name: str
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.name, field="optimization input name")
        for field_name in ("min_shape", "opt_shape", "max_shape"):
            values = tuple(getattr(self, field_name))
            object.__setattr__(self, field_name, values)
            if not values:
                raise ConfigurationError(f"{field_name} must not be empty")
            for index, value in enumerate(values):
                _strict_integer(value, field=f"{field_name}[{index}]", minimum=1)
        if not (len(self.min_shape) == len(self.opt_shape) == len(self.max_shape)):
            raise ConfigurationError("optimization shapes must have equal ranks")
        for minimum, optimum, maximum in zip(
            self.min_shape, self.opt_shape, self.max_shape
        ):
            if not minimum <= optimum <= maximum:
                raise ConfigurationError(
                    "optimization shapes must satisfy min <= opt <= max"
                )

    @classmethod
    def from_dict(cls, value: Any, *, field: str) -> InputShapeRange:
        """Parse one strict input shape range."""

        data = _strict_mapping(value, field=field)
        _require_keys(
            data,
            {"name", "min_shape", "opt_shape", "max_shape"},
            field=field,
        )

        def shape(name: str) -> tuple[int, ...]:
            values = _strict_sequence(data[name], field=f"{field}.{name}")
            return tuple(
                _strict_integer(item, field=f"{field}.{name}[{index}]", minimum=1)
                for index, item in enumerate(values)
            )

        return cls(
            name=_nonempty_string(data["name"], field=f"{field}.name"),
            min_shape=shape("min_shape"),
            opt_shape=shape("opt_shape"),
            max_shape=shape("max_shape"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "name": self.name,
            "min_shape": list(self.min_shape),
            "opt_shape": list(self.opt_shape),
            "max_shape": list(self.max_shape),
        }

    def accepts(self, shape: Sequence[int]) -> bool:
        """Return whether a concrete shape lies inside this inclusive range."""

        concrete = tuple(shape)
        return len(concrete) == len(self.min_shape) and all(
            minimum <= value <= maximum
            for value, minimum, maximum in zip(concrete, self.min_shape, self.max_shape)
        )


@dataclass(frozen=True)
class OptimizationProfile:
    """Named ordered collection of TensorRT input shape ranges."""

    name: str
    inputs: tuple[InputShapeRange, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.name, field="optimization profile name")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not self.inputs:
            raise ConfigurationError("optimization profile inputs must not be empty")
        if not all(isinstance(item, InputShapeRange) for item in self.inputs):
            raise TypeError(
                "optimization profile inputs must be InputShapeRange objects"
            )
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ConfigurationError("optimization profile input names must be unique")

    @classmethod
    def from_dict(cls, value: Any, *, field: str) -> OptimizationProfile:
        """Parse one strict optimization profile."""

        data = _strict_mapping(value, field=field)
        _require_keys(data, {"name", "inputs"}, field=field)
        inputs = _strict_sequence(data["inputs"], field=f"{field}.inputs")
        return cls(
            name=_nonempty_string(data["name"], field=f"{field}.name"),
            inputs=tuple(
                InputShapeRange.from_dict(item, field=f"{field}.inputs[{index}]")
                for index, item in enumerate(inputs)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {"name": self.name, "inputs": [item.to_dict() for item in self.inputs]}

    def require_shapes(self, shapes: Mapping[str, Sequence[int]]) -> None:
        """Reject missing, extra, or out-of-range input shapes."""

        expected = {item.name for item in self.inputs}
        observed = set(shapes)
        if observed != expected:
            raise ConfigurationError(
                f"optimization profile {self.name!r} input mismatch: "
                f"expected {sorted(expected)}, got {sorted(observed)}"
            )
        failures = [
            f"{item.name}={tuple(shapes[item.name])} not in "
            f"[{item.min_shape}, {item.max_shape}]"
            for item in self.inputs
            if not item.accepts(shapes[item.name])
        ]
        if failures:
            raise ConfigurationError(
                f"optimization profile {self.name!r} rejected shapes: "
                + "; ".join(failures)
            )


class EnginePrecision(str, Enum):
    """Builder precision flags allowed while TensorRT selects layer tactics.

    Enabling a flag does not prove that every engine layer executes at that
    precision.  TensorRT may select higher-precision tactics unless layer-level
    constraints and an engine inspector establish otherwise.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass(frozen=True)
class BuildEnvironment:
    """TensorRT/CUDA/GPU identity observed while building or loading an engine."""

    tensorrt_version: str
    cuda_version: str
    gpu_name: str
    gpu_compute_capability: str
    gpu_device_index: int
    gpu_total_memory_bytes: int

    def __post_init__(self) -> None:
        for name in ("tensorrt_version", "cuda_version", "gpu_name"):
            _nonempty_string(getattr(self, name), field=f"environment.{name}")
        if not isinstance(
            self.gpu_compute_capability, str
        ) or not _COMPUTE_CAPABILITY_PATTERN.fullmatch(self.gpu_compute_capability):
            raise ConfigurationError(
                "environment.gpu_compute_capability must look like 'major.minor'"
            )
        _strict_integer(self.gpu_device_index, field="environment.gpu_device_index")
        _strict_integer(
            self.gpu_total_memory_bytes,
            field="environment.gpu_total_memory_bytes",
            minimum=1,
        )

    @classmethod
    def from_dict(cls, value: Any) -> BuildEnvironment:
        """Parse a strict build environment identity."""

        data = _strict_mapping(value, field="environment")
        fields = {
            "tensorrt_version",
            "cuda_version",
            "gpu_name",
            "gpu_compute_capability",
            "gpu_device_index",
            "gpu_total_memory_bytes",
        }
        _require_keys(data, fields, field="environment")
        return cls(
            tensorrt_version=_nonempty_string(
                data["tensorrt_version"], field="environment.tensorrt_version"
            ),
            cuda_version=_nonempty_string(
                data["cuda_version"], field="environment.cuda_version"
            ),
            gpu_name=_nonempty_string(data["gpu_name"], field="environment.gpu_name"),
            gpu_compute_capability=_nonempty_string(
                data["gpu_compute_capability"],
                field="environment.gpu_compute_capability",
            ),
            gpu_device_index=_strict_integer(
                data["gpu_device_index"], field="environment.gpu_device_index"
            ),
            gpu_total_memory_bytes=_strict_integer(
                data["gpu_total_memory_bytes"],
                field="environment.gpu_total_memory_bytes",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "tensorrt_version": self.tensorrt_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_compute_capability": self.gpu_compute_capability,
            "gpu_device_index": self.gpu_device_index,
            "gpu_total_memory_bytes": self.gpu_total_memory_bytes,
        }


@dataclass(frozen=True)
class VerifiedEngine:
    """A manifest whose complete artifact and compatibility gates passed."""

    manifest: EngineManifest
    artifact_root: Path
    engine_path: Path


@dataclass(frozen=True)
class EngineManifest:
    """Exact TensorRT engine build and source-lineage contract.

    Schema v2 calls the serialized value ``enabled_builder_precision_flag``.
    ``precision`` remains the schema-v1 compatibility name.  Neither is a
    claim about actual per-layer engine arithmetic.
    """

    engine: ArtifactDigest
    source_checkpoint: SourceCheckpointIdentity
    build_inputs: tuple[ArtifactDigest, ...]
    build_source_kind: str
    environment: BuildEnvironment
    precision: EnginePrecision
    optimization_profiles: tuple[OptimizationProfile, ...]
    workspace_size_bytes: int
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _strict_integer(self.schema_version, field="schema_version", minimum=1)
        if self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ConfigurationError(
                f"unsupported engine manifest schema_version={self.schema_version}"
            )
        if not isinstance(self.engine, ArtifactDigest):
            raise TypeError("engine must be an ArtifactDigest")
        if not isinstance(self.source_checkpoint, SourceCheckpointIdentity):
            raise TypeError("source_checkpoint must be SourceCheckpointIdentity")
        object.__setattr__(self, "build_inputs", tuple(self.build_inputs))
        if not self.build_inputs:
            raise ConfigurationError("build_inputs must not be empty")
        if not all(isinstance(item, ArtifactDigest) for item in self.build_inputs):
            raise TypeError("build_inputs must contain ArtifactDigest objects")
        if self.build_source_kind not in {"onnx", "network_callback"}:
            raise ConfigurationError(
                "build_source_kind must be 'onnx' or 'network_callback'"
            )
        if not isinstance(self.environment, BuildEnvironment):
            raise TypeError("environment must be BuildEnvironment")
        if not isinstance(self.precision, EnginePrecision):
            try:
                object.__setattr__(self, "precision", EnginePrecision(self.precision))
            except (TypeError, ValueError) as error:
                raise ConfigurationError(
                    f"unsupported TensorRT builder precision flag: {self.precision!r}"
                ) from error
        object.__setattr__(
            self, "optimization_profiles", tuple(self.optimization_profiles)
        )
        if not self.optimization_profiles:
            raise ConfigurationError("optimization_profiles must not be empty")
        if not all(
            isinstance(item, OptimizationProfile) for item in self.optimization_profiles
        ):
            raise TypeError(
                "optimization_profiles must contain OptimizationProfile objects"
            )
        profile_names = [profile.name for profile in self.optimization_profiles]
        if len(profile_names) != len(set(profile_names)):
            raise ConfigurationError("optimization profile names must be unique")
        profile_inputs = [
            tuple(item.name for item in profile.inputs)
            for profile in self.optimization_profiles
        ]
        if any(names != profile_inputs[0] for names in profile_inputs[1:]):
            raise ConfigurationError(
                "all optimization profiles must declare the same ordered inputs"
            )
        _strict_integer(
            self.workspace_size_bytes,
            field="workspace_size_bytes",
            minimum=1,
        )
        all_paths = [
            self.engine.path,
            *(item.path for item in self.build_inputs),
            *(item.path for item in self.source_checkpoint.files),
            self.source_checkpoint.norm_stats.path,
            *(item.path for item in self.source_checkpoint.transforms),
        ]
        if len(all_paths) != len(set(all_paths)):
            raise ConfigurationError("all manifest artifact paths must be unique")

    @classmethod
    def from_dict(cls, value: Any) -> EngineManifest:
        """Parse a manifest object with no ignored or coerced fields."""

        data = _strict_mapping(value, field="manifest")
        if "schema_version" not in data:
            raise ConfigurationError("manifest: missing=schema_version")
        schema_version = _strict_integer(
            data["schema_version"], field="schema_version", minimum=1
        )
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ConfigurationError(
                f"unsupported engine manifest schema_version={schema_version}"
            )
        precision_field = (
            "precision"
            if schema_version == _LEGACY_SCHEMA_VERSION
            else "enabled_builder_precision_flag"
        )
        fields = {
            "schema_version",
            "engine",
            "source_checkpoint",
            "build_inputs",
            "build_source_kind",
            "environment",
            precision_field,
            "optimization_profiles",
            "workspace_size_bytes",
        }
        _require_keys(data, fields, field="manifest")
        build_inputs = _strict_sequence(data["build_inputs"], field="build_inputs")
        profiles = _strict_sequence(
            data["optimization_profiles"], field="optimization_profiles"
        )
        precision_value = _nonempty_string(data[precision_field], field=precision_field)
        try:
            precision = EnginePrecision(precision_value)
        except ValueError as error:
            raise ConfigurationError(
                f"unsupported TensorRT builder precision flag: {precision_value!r}"
            ) from error
        return cls(
            schema_version=schema_version,
            engine=ArtifactDigest.from_dict(data["engine"], field="engine"),
            source_checkpoint=SourceCheckpointIdentity.from_dict(
                data["source_checkpoint"]
            ),
            build_inputs=tuple(
                ArtifactDigest.from_dict(item, field=f"build_inputs[{index}]")
                for index, item in enumerate(build_inputs)
            ),
            build_source_kind=_nonempty_string(
                data["build_source_kind"], field="build_source_kind"
            ),
            environment=BuildEnvironment.from_dict(data["environment"]),
            precision=precision,
            optimization_profiles=tuple(
                OptimizationProfile.from_dict(
                    item, field=f"optimization_profiles[{index}]"
                )
                for index, item in enumerate(profiles)
            ),
            workspace_size_bytes=_strict_integer(
                data["workspace_size_bytes"],
                field="workspace_size_bytes",
                minimum=1,
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> EngineManifest:
        """Parse strict JSON, rejecting duplicate keys and non-finite constants."""

        return cls.from_dict(_load_strict_json(text))

    @classmethod
    def from_path(cls, path: str | Path) -> EngineManifest:
        """Load a strict manifest from disk."""

        manifest_path = Path(path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        try:
            return cls.from_json(manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ConfigurationError(
                f"could not read engine manifest {manifest_path}: {error}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible manifest representation."""

        payload = {
            "schema_version": self.schema_version,
            "engine": self.engine.to_dict(),
            "source_checkpoint": self.source_checkpoint.to_dict(),
            "build_inputs": [item.to_dict() for item in self.build_inputs],
            "build_source_kind": self.build_source_kind,
            "environment": self.environment.to_dict(),
            "optimization_profiles": [
                profile.to_dict() for profile in self.optimization_profiles
            ],
            "workspace_size_bytes": self.workspace_size_bytes,
        }
        precision_field = (
            "precision"
            if self.schema_version == _LEGACY_SCHEMA_VERSION
            else "enabled_builder_precision_flag"
        )
        payload[precision_field] = self.precision.value
        return payload

    @property
    def enabled_builder_precision_flag(self) -> EnginePrecision:
        """Return the builder flag permitted during TensorRT tactic selection."""

        return self.precision

    @property
    def enabled_precision_flag(self) -> EnginePrecision:
        """Compatibility alias for :attr:`enabled_builder_precision_flag`."""

        return self.enabled_builder_precision_flag

    def to_json(self) -> str:
        """Serialize deterministically for review and source control."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically write the manifest without silently replacing evidence."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(self.to_json())
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination

    def profile(self, name: str) -> OptimizationProfile:
        """Return one named optimization profile or reject the request."""

        matches = [
            profile for profile in self.optimization_profiles if profile.name == name
        ]
        if len(matches) != 1:
            raise ConfigurationError(f"unknown optimization profile: {name!r}")
        return matches[0]

    def require_profile(
        self, name: str, shapes: Mapping[str, Sequence[int]]
    ) -> OptimizationProfile:
        """Require one named profile to accept the exact runtime input set."""

        profile = self.profile(name)
        profile.require_shapes(shapes)
        return profile

    def verify(
        self,
        artifact_root: str | Path,
        *,
        expected_source_checkpoint: SourceCheckpointIdentity | None = None,
        expected_environment: BuildEnvironment | None = None,
        expected_precision: EnginePrecision | str | None = None,
        expected_profiles: Sequence[OptimizationProfile] | None = None,
    ) -> VerifiedEngine:
        """Verify artifacts and every caller-supplied compatibility expectation."""

        root = Path(artifact_root).expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"artifact root is not a directory: {root}")
        engine_path = self.engine.verify(root)
        self.source_checkpoint.verify(root)
        for artifact in self.build_inputs:
            artifact.verify(root)
        if (
            expected_source_checkpoint is not None
            and self.source_checkpoint != expected_source_checkpoint
        ):
            raise ConfigurationError("source checkpoint identity mismatch")
        if (
            expected_environment is not None
            and self.environment != expected_environment
        ):
            raise ConfigurationError(
                "TensorRT/CUDA/GPU environment mismatch: "
                f"built={self.environment.to_dict()}, "
                f"runtime={expected_environment.to_dict()}"
            )
        if expected_precision is not None:
            try:
                precision = EnginePrecision(expected_precision)
            except (TypeError, ValueError) as error:
                raise ConfigurationError(
                    "unsupported expected TensorRT builder precision flag: "
                    f"{expected_precision!r}"
                ) from error
            if self.precision is not precision:
                raise ConfigurationError(
                    "enabled TensorRT builder precision flag mismatch: "
                    f"built={self.precision.value}, "
                    f"expected={precision.value}"
                )
        if expected_profiles is not None and self.optimization_profiles != tuple(
            expected_profiles
        ):
            raise ConfigurationError("optimization profile set mismatch")
        return VerifiedEngine(
            manifest=self, artifact_root=root, engine_path=engine_path
        )


@dataclass(frozen=True)
class TensorRTBuildSpec:
    """Inputs required to build and bind a TensorRT engine manifest.

    ``precision`` enables an allowed TensorRT builder precision flag; it does
    not assert the selected precision of every serialized-engine layer.
    """

    source_checkpoint: SourceCheckpointIdentity
    build_inputs: tuple[ArtifactDigest, ...]
    precision: EnginePrecision
    optimization_profiles: tuple[OptimizationProfile, ...]
    workspace_size_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        if not isinstance(self.source_checkpoint, SourceCheckpointIdentity):
            raise TypeError("source_checkpoint must be SourceCheckpointIdentity")
        object.__setattr__(self, "build_inputs", tuple(self.build_inputs))
        object.__setattr__(
            self, "optimization_profiles", tuple(self.optimization_profiles)
        )
        if not self.build_inputs:
            raise ConfigurationError("build_inputs must not be empty")
        if not all(isinstance(item, ArtifactDigest) for item in self.build_inputs):
            raise TypeError("build_inputs must contain ArtifactDigest objects")
        if not self.optimization_profiles:
            raise ConfigurationError("optimization_profiles must not be empty")
        if not all(
            isinstance(item, OptimizationProfile) for item in self.optimization_profiles
        ):
            raise TypeError(
                "optimization_profiles must contain OptimizationProfile objects"
            )
        profile_names = [profile.name for profile in self.optimization_profiles]
        if len(profile_names) != len(set(profile_names)):
            raise ConfigurationError("optimization profile names must be unique")
        profile_inputs = [
            tuple(item.name for item in profile.inputs)
            for profile in self.optimization_profiles
        ]
        if any(names != profile_inputs[0] for names in profile_inputs[1:]):
            raise ConfigurationError(
                "all optimization profiles must declare the same ordered inputs"
            )
        artifact_paths = [
            *(item.path for item in self.source_checkpoint.files),
            self.source_checkpoint.norm_stats.path,
            *(item.path for item in self.source_checkpoint.transforms),
            *(item.path for item in self.build_inputs),
        ]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ConfigurationError("build/source artifact paths must be unique")
        if not isinstance(self.precision, EnginePrecision):
            try:
                object.__setattr__(self, "precision", EnginePrecision(self.precision))
            except (TypeError, ValueError) as error:
                raise ConfigurationError(
                    f"unsupported TensorRT builder precision flag: {self.precision!r}"
                ) from error
        _strict_integer(
            self.workspace_size_bytes,
            field="workspace_size_bytes",
            minimum=1,
        )


NetworkPopulateCallback = Callable[[Any, Any, Any], Any]


def _import_optional(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise OptionalDependencyError(message) from error


def detect_build_environment(
    *, trt_module: Any | None = None, torch_module: Any | None = None
) -> BuildEnvironment:
    """Observe the active TensorRT, CUDA toolkit, and CUDA-device identity."""

    trt = trt_module or _import_optional(
        "tensorrt", "TensorRT environment detection requires the tensorrt package"
    )
    torch = torch_module or _import_optional(
        "torch", "TensorRT environment detection requires a CUDA-enabled torch package"
    )
    if not torch.cuda.is_available():
        raise OptionalDependencyError(
            "TensorRT environment detection requires an available CUDA device"
        )
    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version:
        raise OptionalDependencyError("torch does not report its CUDA toolkit version")
    return BuildEnvironment(
        tensorrt_version=_nonempty_string(
            getattr(trt, "__version__", None), field="tensorrt.__version__"
        ),
        cuda_version=cuda_version,
        gpu_name=str(properties.name),
        gpu_compute_capability=f"{int(properties.major)}.{int(properties.minor)}",
        gpu_device_index=device_index,
        gpu_total_memory_bytes=int(properties.total_memory),
    )


def _configure_builder(
    trt: Any,
    builder: Any,
    config: Any,
    spec: TensorRTBuildSpec,
) -> None:
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, spec.workspace_size_bytes
        )
    else:
        raise RuntimeError("TensorRT 10 set_memory_pool_limit API is required")

    flag_names = {
        EnginePrecision.FP16: "FP16",
        EnginePrecision.BF16: "BF16",
        EnginePrecision.INT8: "INT8",
    }
    # Builder flags permit tactic selection at a precision; they do not prove
    # that every resulting engine layer runs at that precision.
    if spec.precision in flag_names:
        flag_name = flag_names[spec.precision]
        if not hasattr(trt.BuilderFlag, flag_name):
            raise RuntimeError(
                f"installed TensorRT does not expose BuilderFlag.{flag_name}"
            )
        config.set_flag(getattr(trt.BuilderFlag, flag_name))

    for profile_spec in spec.optimization_profiles:
        profile = builder.create_optimization_profile()
        for input_spec in profile_spec.inputs:
            accepted = profile.set_shape(
                input_spec.name,
                input_spec.min_shape,
                input_spec.opt_shape,
                input_spec.max_shape,
            )
            if accepted is False:
                raise RuntimeError(
                    f"TensorRT rejected optimization shape for {input_spec.name!r}"
                )
        accepted = config.add_optimization_profile(profile)
        if accepted is False or accepted == -1:
            raise RuntimeError(
                f"TensorRT rejected optimization profile {profile_spec.name!r}"
            )


def _write_engine_bytes(
    destination: Path, serialized: bytes, *, overwrite: bool
) -> None:
    if not serialized:
        raise RuntimeError("TensorRT returned an empty serialized engine")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_tensorrt_engine(
    artifact_root: str | Path,
    *,
    engine_path: str,
    manifest_path: str,
    spec: TensorRTBuildSpec,
    onnx_path: str | None = None,
    network_populator: NetworkPopulateCallback | None = None,
    observed_environment: BuildEnvironment | None = None,
    trt_module: Any | None = None,
    logger_severity: Any | None = None,
    overwrite: bool = False,
) -> VerifiedEngine:
    """Build an engine from ONNX or an explicit network-population callback.

    Exactly one of ``onnx_path`` and ``network_populator`` is required.  The
    callback receives ``(builder, network, logger)`` and must populate the
    network; returning ``False`` rejects the build.  It is responsible for
    retaining any parser/export objects until it returns.  ``build_inputs`` in
    ``spec`` must bind every exported artifact used by either path.
    """

    if (onnx_path is None) == (network_populator is None):
        raise ConfigurationError(
            "provide exactly one of onnx_path or network_populator"
        )
    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"artifact root is not a directory: {root}")
    engine_relative = _canonical_relative_path(engine_path, field="engine_path")
    manifest_relative = _canonical_relative_path(manifest_path, field="manifest_path")
    engine_destination = _resolve_artifact(root, engine_relative)
    manifest_destination = _resolve_artifact(root, manifest_relative)
    if engine_destination == manifest_destination:
        raise ConfigurationError("engine_path and manifest_path must be different")
    protected_paths = {
        *(artifact.path for artifact in spec.source_checkpoint.files),
        spec.source_checkpoint.norm_stats.path,
        *(artifact.path for artifact in spec.source_checkpoint.transforms),
        *(artifact.path for artifact in spec.build_inputs),
    }
    collisions = {engine_relative, manifest_relative} & protected_paths
    if collisions:
        raise ConfigurationError(
            "engine or manifest output collides with a bound source artifact: "
            + ", ".join(sorted(collisions))
        )
    if not overwrite:
        existing_outputs = [
            path for path in (engine_destination, manifest_destination) if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(existing_outputs[0])

    spec.source_checkpoint.verify(root)
    for artifact in spec.build_inputs:
        artifact.verify(root)
    if onnx_path is not None:
        onnx_relative = _canonical_relative_path(onnx_path, field="onnx_path")
        if onnx_relative not in {artifact.path for artifact in spec.build_inputs}:
            raise ConfigurationError(
                "onnx_path must be present in the SHA-bound build_inputs"
            )
        onnx_candidate = _resolve_artifact(root, onnx_relative)
        build_source_kind = "onnx"
    else:
        onnx_candidate = None
        build_source_kind = "network_callback"
        if not callable(network_populator):
            raise TypeError("network_populator must be callable")

    trt = trt_module or _import_optional(
        "tensorrt", "building a TensorRT engine requires the tensorrt package"
    )
    environment = observed_environment or detect_build_environment(trt_module=trt)
    severity = trt.Logger.WARNING if logger_severity is None else logger_severity
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    config = builder.create_builder_config()
    parser_owner: Any | None = None
    if onnx_candidate is not None:
        parser_owner = trt.OnnxParser(network, logger)
        if not parser_owner.parse(onnx_candidate.read_bytes()):
            errors = [
                str(parser_owner.get_error(index))
                for index in range(int(parser_owner.num_errors))
            ]
            raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(errors))
    else:
        assert network_populator is not None
        parser_owner = network_populator(builder, network, logger)
        if parser_owner is False:
            raise RuntimeError("network_populator rejected the TensorRT network")

    _configure_builder(trt, builder, config, spec)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")
    _write_engine_bytes(engine_destination, bytes(serialized), overwrite=overwrite)

    manifest = EngineManifest(
        engine=ArtifactDigest.from_path(root, engine_relative),
        source_checkpoint=spec.source_checkpoint,
        build_inputs=spec.build_inputs,
        build_source_kind=build_source_kind,
        environment=environment,
        precision=spec.precision,
        optimization_profiles=spec.optimization_profiles,
        workspace_size_bytes=spec.workspace_size_bytes,
    )
    try:
        manifest.write(manifest_destination, overwrite=overwrite)
    except BaseException:
        if not overwrite:
            engine_destination.unlink(missing_ok=True)
        raise
    return manifest.verify(
        root,
        expected_source_checkpoint=spec.source_checkpoint,
        expected_environment=environment,
        expected_precision=spec.precision,
        expected_profiles=spec.optimization_profiles,
    )


def _torch_dtype(trt_dtype: Any, torch: Any) -> Any:
    name = str(trt_dtype).lower().replace("datatype.", "")
    aliases = {
        "float": "float32",
        "half": "float16",
        "bool": "bool",
        "int8": "int8",
        "int32": "int32",
        "int64": "int64",
        "uint8": "uint8",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "fp8": "float8_e4m3fn",
    }
    attribute = aliases.get(name, name)
    if not hasattr(torch, attribute):
        raise RuntimeError(f"unsupported TensorRT tensor dtype: {trt_dtype}")
    return getattr(torch, attribute)


def _verify_deserialized_profiles(
    engine: Any, trt: Any, manifest: EngineManifest
) -> None:
    if not hasattr(engine, "num_optimization_profiles") or not hasattr(
        engine, "get_tensor_profile_shape"
    ):
        raise RuntimeError("TensorRT 10 engine profile introspection APIs are required")
    observed_profile_count = int(engine.num_optimization_profiles)
    expected_profile_count = len(manifest.optimization_profiles)
    if observed_profile_count != expected_profile_count:
        raise ConfigurationError(
            "deserialized engine optimization profile count mismatch: "
            f"engine={observed_profile_count}, manifest={expected_profile_count}"
        )
    tensor_names = [
        engine.get_tensor_name(index) for index in range(int(engine.num_io_tensors))
    ]
    engine_inputs = {
        name
        for name in tensor_names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    }
    manifest_inputs = {item.name for item in manifest.optimization_profiles[0].inputs}
    if engine_inputs != manifest_inputs:
        raise ConfigurationError(
            f"deserialized engine input mismatch: engine={sorted(engine_inputs)}, "
            f"manifest={sorted(manifest_inputs)}"
        )
    for profile_index, profile in enumerate(manifest.optimization_profiles):
        for input_spec in profile.inputs:
            observed = engine.get_tensor_profile_shape(input_spec.name, profile_index)
            try:
                observed_shapes = tuple(
                    tuple(int(value) for value in shape) for shape in observed
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "TensorRT returned an invalid optimization profile shape for "
                    f"{input_spec.name!r}"
                ) from error
            expected_shapes = (
                input_spec.min_shape,
                input_spec.opt_shape,
                input_spec.max_shape,
            )
            if observed_shapes != expected_shapes:
                raise ConfigurationError(
                    "deserialized engine optimization profile mismatch for "
                    f"profile={profile.name!r}, input={input_spec.name!r}: "
                    f"engine={observed_shapes}, manifest={expected_shapes}"
                )


def _cuda_device_scope(torch_module: Any, device_index: int) -> Any:
    """Enter one CUDA device when the injected Torch API exposes the guard."""

    device_guard = getattr(getattr(torch_module, "cuda", None), "device", None)
    if callable(device_guard):
        return device_guard(device_index)
    # CPU-only fake modules used by unit tests do not need a CUDA guard.
    return nullcontext()


def _cuda_device_index(torch_module: Any, device: Any) -> int:
    """Resolve an explicit CUDA device and reject CPU/unknown device objects."""

    if isinstance(device, str):
        match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
        if match is None:
            raise ConfigurationError(f"expected a CUDA device, got {device!r}")
        if match.group(1) is not None:
            return int(match.group(1))
        current_device = getattr(
            getattr(torch_module, "cuda", None), "current_device", None
        )
        if not callable(current_device):
            raise ConfigurationError("unindexed CUDA device cannot be resolved")
        return int(current_device())

    if getattr(device, "type", None) != "cuda":
        raise ConfigurationError(f"expected a CUDA device, got {device!r}")
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    current_device = getattr(
        getattr(torch_module, "cuda", None), "current_device", None
    )
    if not callable(current_device):
        raise ConfigurationError("unindexed CUDA device cannot be resolved")
    return int(current_device())


class TensorRTV3Runtime:
    """Manifest-gated TensorRT 10 ``execute_async_v3`` runtime.

    Inputs must be contiguous CUDA torch tensors on the exact GPU recorded in
    the verified manifest.  Their storage pointers are bound with
    ``set_tensor_address``; output CUDA tensors are allocated only after all
    dynamic input shapes have been accepted.  Returned tensors are
    asynchronous unless ``synchronize=True`` is requested.

    Execution contexts are retained independently for each host-thread/CUDA-
    stream pair.  TensorRT address mutation is therefore never shared across
    concurrent callers or distinct streams, while same-stream replays remain
    ordered by CUDA without forcing a host synchronization.
    """

    def __init__(
        self,
        verified: VerifiedEngine,
        profile_name: str,
        *,
        trt_module: Any,
        torch_module: Any,
        logger: Any,
        runtime: Any,
        engine: Any,
        context: Any,
    ) -> None:
        self.verified = verified
        self.profile_name = profile_name
        self.trt = trt_module
        self.torch = torch_module
        self.logger = logger
        self.runtime = runtime
        self.engine = engine
        self.device_index = verified.manifest.environment.gpu_device_index
        self._context_lock = threading.Lock()
        self._contexts: dict[tuple[int, int], Any] = {}
        self._thread_tokens: WeakKeyDictionary[threading.Thread, int] = (
            WeakKeyDictionary()
        )
        self._next_thread_token = 0
        self._unclaimed_context: Any | None = context
        self._last_context = context

    @property
    def context(self) -> Any:
        """Return the last selected context for diagnostics and compatibility."""

        with self._context_lock:
            return self._last_context

    def _create_execution_context(self) -> Any:
        with _cuda_device_scope(self.torch, self.device_index):
            context = self.engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        if not hasattr(context, "execute_async_v3"):
            raise RuntimeError("TensorRT 10 execute_async_v3 API is required")
        return context

    def _context_for_stream(self, stream_handle: int) -> Any:
        with self._context_lock:
            thread = threading.current_thread()
            thread_token = self._thread_tokens.get(thread)
            if thread_token is None:
                self._next_thread_token += 1
                thread_token = self._next_thread_token
                self._thread_tokens[thread] = thread_token
            # Monotonic tokens cannot be reused after a short-lived thread
            # exits while its asynchronous enqueue is still in flight.
            key = (thread_token, stream_handle)
            context = self._contexts.get(key)
            if context is None:
                if self._unclaimed_context is not None:
                    context = self._unclaimed_context
                    self._unclaimed_context = None
                else:
                    context = self._create_execution_context()
                self._contexts[key] = context
            self._last_context = context
            return context

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        artifact_root: str | Path,
        *,
        profile_name: str,
        expected_source_checkpoint: SourceCheckpointIdentity | None = None,
        expected_precision: EnginePrecision | str | None = None,
        expected_profiles: Sequence[OptimizationProfile] | None = None,
        observed_environment: BuildEnvironment | None = None,
        trt_module: Any | None = None,
        torch_module: Any | None = None,
        logger_severity: Any | None = None,
    ) -> TensorRTV3Runtime:
        """Verify lineage and environment before deserializing one engine."""

        manifest = EngineManifest.from_path(manifest_path)
        trt = trt_module or _import_optional(
            "tensorrt", "loading a TensorRT engine requires the tensorrt package"
        )
        torch = torch_module or _import_optional(
            "torch", "TensorRT v3 execution requires a CUDA-enabled torch package"
        )
        environment = observed_environment or detect_build_environment(
            trt_module=trt, torch_module=torch
        )
        verified = manifest.verify(
            artifact_root,
            expected_source_checkpoint=expected_source_checkpoint,
            expected_environment=environment,
            expected_precision=expected_precision,
            expected_profiles=expected_profiles,
        )
        manifest.profile(profile_name)
        severity = trt.Logger.WARNING if logger_severity is None else logger_severity
        logger = trt.Logger(severity)
        runtime_device_index = manifest.environment.gpu_device_index
        with _cuda_device_scope(torch, runtime_device_index):
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(verified.engine_path.read_bytes())
            if engine is None:
                raise RuntimeError(
                    f"TensorRT could not deserialize engine {verified.engine_path}"
                )
            _verify_deserialized_profiles(engine, trt, manifest)
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError("TensorRT could not create an execution context")
            if not hasattr(context, "execute_async_v3"):
                raise RuntimeError("TensorRT 10 execute_async_v3 API is required")
        return cls(
            verified,
            profile_name,
            trt_module=trt,
            torch_module=torch,
            logger=logger,
            runtime=runtime,
            engine=engine,
            context=context,
        )

    def execute(
        self, inputs: Mapping[str, Any], *, synchronize: bool = False
    ) -> dict[str, Any]:
        """Bind CUDA pointers, enqueue ``execute_async_v3``, and return outputs."""

        if not inputs:
            raise ConfigurationError("TensorRT inputs must not be empty")
        shapes = {
            name: tuple(int(value) for value in tensor.shape)
            for name, tensor in inputs.items()
        }
        profile = self.verified.manifest.require_profile(self.profile_name, shapes)
        profile_index = self.verified.manifest.optimization_profiles.index(profile)

        first_tensor = next(iter(inputs.values()))
        device = getattr(first_tensor, "device", None)
        expected_device_index = self.device_index
        for name, tensor in inputs.items():
            if not bool(getattr(tensor, "is_cuda", False)):
                raise ConfigurationError(
                    f"TensorRT input {name!r} is not a CUDA tensor"
                )
            input_device_index = _cuda_device_index(
                self.torch, getattr(tensor, "device", None)
            )
            if input_device_index != expected_device_index:
                raise ConfigurationError(
                    f"TensorRT input {name!r} is on cuda:{input_device_index}; "
                    "the verified manifest requires "
                    f"cuda:{expected_device_index}"
                )

        with _cuda_device_scope(self.torch, expected_device_index):
            stream = self.torch.cuda.current_stream(device=device)
            stream_handle = int(stream.cuda_stream)
            context = self._context_for_stream(stream_handle)
            if hasattr(context, "set_optimization_profile_async"):
                accepted = context.set_optimization_profile_async(
                    profile_index, stream_handle
                )
                if accepted is False:
                    raise RuntimeError(
                        f"TensorRT rejected optimization profile {self.profile_name!r}"
                    )
            elif profile_index != 0:
                raise RuntimeError(
                    "TensorRT context cannot select the requested optimization profile"
                )

            tensor_names = [
                self.engine.get_tensor_name(index)
                for index in range(int(self.engine.num_io_tensors))
            ]
            input_names = {
                name
                for name in tensor_names
                if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
            }
            if input_names != set(inputs):
                raise ConfigurationError(
                    f"engine input mismatch: expected {sorted(input_names)}, "
                    f"got {sorted(inputs)}"
                )

            for name, tensor in inputs.items():
                if not tensor.is_contiguous():
                    raise ConfigurationError(
                        f"TensorRT input {name!r} must be contiguous"
                    )
                expected_dtype = _torch_dtype(
                    self.engine.get_tensor_dtype(name), self.torch
                )
                if tensor.dtype != expected_dtype:
                    raise ConfigurationError(
                        f"TensorRT input {name!r} dtype mismatch: "
                        f"engine={expected_dtype}, input={tensor.dtype}"
                    )
                accepted = context.set_input_shape(name, tuple(tensor.shape))
                if accepted is False:
                    raise RuntimeError(f"TensorRT rejected input shape for {name!r}")
                accepted = context.set_tensor_address(name, int(tensor.data_ptr()))
                if accepted is False:
                    raise RuntimeError(f"TensorRT rejected input address for {name!r}")

            outputs: dict[str, Any] = {}
            for name in tensor_names:
                if name in input_names:
                    continue
                shape = tuple(int(value) for value in context.get_tensor_shape(name))
                if not shape or any(value <= 0 for value in shape):
                    raise RuntimeError(
                        f"TensorRT output {name!r} retained an unresolved shape {shape}"
                    )
                dtype = _torch_dtype(self.engine.get_tensor_dtype(name), self.torch)
                output = self.torch.empty(shape, dtype=dtype, device=device)
                accepted = context.set_tensor_address(name, int(output.data_ptr()))
                if accepted is False:
                    raise RuntimeError(f"TensorRT rejected output address for {name!r}")
                outputs[name] = output
            if not outputs:
                raise RuntimeError("TensorRT engine does not expose an output tensor")
            accepted = context.execute_async_v3(stream_handle)
            if accepted is False:
                raise RuntimeError("TensorRT execute_async_v3 failed")
            if synchronize:
                stream.synchronize()
            return outputs


__all__ = [
    "ArtifactDigest",
    "BuildEnvironment",
    "EngineManifest",
    "EnginePrecision",
    "InputShapeRange",
    "NetworkPopulateCallback",
    "OptimizationProfile",
    "SourceCheckpointIdentity",
    "TensorRTBuildSpec",
    "TensorRTV3Runtime",
    "VerifiedEngine",
    "build_tensorrt_engine",
    "detect_build_environment",
]
