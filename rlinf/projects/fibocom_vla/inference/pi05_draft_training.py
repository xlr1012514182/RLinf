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

"""Evidence-bound materialization and training for the pi0.5 Draft head.

This module deliberately separates three authorities:

* a source adapter may provide only raw :class:`Observation` objects and
  episode/frame lineage;
* :class:`OpenPITeacherOwner` owns all checkpoint-bound OpenPI preprocessing,
  prefix extraction, and teacher sampling;
* candidate export accepts only a hash-verified run and recomputes validation
  metrics from that run's weights.

The public Dexmal Draft is never treated as a pi0.5 policy.  The standalone
CLI loads it through the pinned registry, with ``purpose="warm_start"``, and
then derives a new 32-dimensional target head.  This module cannot sign or
approve a candidate for production.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from ..contracts import Observation
from ..errors import ConfigurationError, ShapeMismatchError
from ..openpi_adapter import RLinfOpenPiChunkPolicy
from .openpi_speculative import OpenPIParallelVerifier, VerifiedOpenPITarget
from .pi05_draft import (
    PI05_ACTION_HORIZON,
    PI05_MODEL_ACTION_DIM,
    PI05_MODEL_STATE_DIM,
    PI05_PREFIX_EMBEDDING_DIM,
    Pi05DraftChunkHead,
    Pi05DraftLossConfig,
    Pi05DraftTargetContract,
    Pi05DraftTrainingRecord,
    WarmStartLineage,
    build_pi05_draft_loss_mask,
    export_pi05_draft_candidate,
    pi05_draft_distillation_loss,
)

CACHE_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
CACHE_KIND = "fibocom_pi05_draft_teacher_cache"
RUN_KIND = "fibocom_pi05_draft_training_run"
_CACHE_PROOF = object()
_RUN_PROOF = object()
_SAMPLE_TENSOR_KEYS = frozenset(
    {
        "prefix_embs",
        "prefix_pad_masks",
        "prefix_att_masks",
        "robot_state",
        "last_actions",
        "teacher_actions",
        "valid_steps",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"value is not canonical JSON: {error}") from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a SHA-256 digest")
    digest = value.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ConfigurationError(f"{name} must be a SHA-256 digest")
    return digest


def _require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _require_revision(value: Any, name: str) -> str:
    revision = _require_nonempty(value, name).lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ConfigurationError(f"{name} must be a 40-character commit revision")
    return revision


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise ConfigurationError(
            f"{name} keys mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _safe_child(root: Path, relative: Any, name: str) -> Path:
    raw = _require_nonempty(relative, name).replace("\\", "/")
    candidate = Path(raw)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ConfigurationError(f"{name} must stay inside the artifact directory")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ConfigurationError(f"{name} escapes the artifact directory") from error
    return resolved


def _atomic_json(path: Path, value: Any) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _safe_torch_load(path: Path, *, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as error:
        raise ConfigurationError(
            "training checkpoint restore requires Torch with weights_only=True"
        ) from error


def _state_schema(module: nn.Module) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name, tensor in sorted(module.state_dict().items()):
        schema[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
    return schema


def _model_state_content_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if tensor.device.type == "meta":
            raise ConfigurationError("cannot hash a meta-device training model")
        header = _canonical_json_bytes(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = (
            tensor.detach().to(device="cpu").contiguous().reshape(-1).view(torch.uint8)
        )
        digest.update(raw.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _save_model_state(path: Path, module: nn.Module) -> None:
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in module.state_dict().items()
    }
    save_file(state, str(path), metadata={"production_authorized": "false"})


def _state_file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(load_file(str(path), device="cpu").items()):
        header = _canonical_json_bytes(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(
            tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def _load_model_state(path: Path, module: nn.Module, device: torch.device) -> None:
    state = load_file(str(path), device=str(device))
    expected = set(module.state_dict())
    if set(state) != expected:
        raise ConfigurationError(
            "training-run model schema changed: "
            f"missing={sorted(expected - set(state))}, "
            f"unexpected={sorted(set(state) - expected)}"
        )
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ConfigurationError("strict training-run model load was incompatible")


@dataclass(frozen=True, kw_only=True)
class DraftSourceSample:
    """One raw source item; no model-space teacher fields are accepted."""

    observation: Observation
    episode_id: str
    frame_index: int
    valid_steps: int = PI05_ACTION_HORIZON

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise TypeError("source observation must be a raw Observation")
        object.__setattr__(
            self, "episode_id", _require_nonempty(self.episode_id, "episode_id")
        )
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ConfigurationError("frame_index must be a non-negative integer")
        if (
            isinstance(self.valid_steps, bool)
            or not isinstance(self.valid_steps, int)
            or not 1 <= self.valid_steps <= PI05_ACTION_HORIZON
        ):
            raise ConfigurationError("valid_steps must be an integer in [1, 50]")


@runtime_checkable
class RawObservationSource(Protocol):
    """Adapter boundary for real robot/dataset observations."""

    def iter_samples(self) -> Iterable[DraftSourceSample]:
        """Yield raw observations with immutable episode/frame lineage."""


def load_source_adapter(
    specification: str,
    *,
    source_root: Path,
    source_manifest_path: Path,
    options: Mapping[str, Any] | None = None,
) -> RawObservationSource:
    """Load a reviewed ``module:function`` raw-observation source factory."""

    if specification.count(":") != 1:
        raise ConfigurationError("source adapter must use module:function syntax")
    module_name, function_name = specification.split(":", 1)
    if not module_name.strip() or not function_name.strip():
        raise ConfigurationError("source adapter must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ConfigurationError("source adapter factory is not callable")
    adapter = factory(
        source_root=source_root.expanduser().resolve(),
        source_manifest_path=source_manifest_path.expanduser().resolve(),
        options=dict(options or {}),
    )
    if not isinstance(adapter, RawObservationSource):
        raise ConfigurationError(
            "source adapter must expose iter_samples() and may return only "
            "DraftSourceSample values"
        )
    return adapter


@dataclass(frozen=True)
class TeacherMaterialization:
    """Target-owned model-space tensors for one raw observation."""

    prefix_embs: torch.Tensor
    prefix_pad_masks: torch.Tensor
    prefix_att_masks: torch.Tensor
    robot_state: torch.Tensor
    teacher_actions: torch.Tensor


class OpenPITeacherOwner:
    """Own all preprocessing and teacher generation for one verified target."""

    def __init__(
        self,
        verified_target: VerifiedOpenPITarget,
        policy: RLinfOpenPiChunkPolicy,
    ) -> None:
        if not isinstance(verified_target, VerifiedOpenPITarget):
            raise TypeError("verified_target must be a VerifiedOpenPITarget")
        if not isinstance(policy, RLinfOpenPiChunkPolicy):
            raise TypeError("policy must be an RLinfOpenPiChunkPolicy")
        verified_target.validate_runtime_model()
        if policy.model is not verified_target.model:
            raise ConfigurationError(
                "teacher policy and VerifiedOpenPITarget must own the same model"
            )
        self.verified_target = verified_target
        self.policy = policy
        self.target_contract = Pi05DraftTargetContract.from_verified_openpi(
            verified_target
        )
        self._verifier = OpenPIParallelVerifier(verified_target)
        self._active_episode: str | None = None

    def _activate(self, episode_id: str) -> None:
        if self._active_episode == episode_id:
            return
        if self._active_episode is not None:
            self._verifier.reset_episode(self._active_episode)
        self._verifier.begin_episode(episode_id)
        self._active_episode = episode_id

    def close(self) -> None:
        """Invalidate any model prefix/KV cache owned by this materializer."""

        if self._active_episode is not None:
            self._verifier.reset_episode(self._active_episode)
            self._active_episode = None

    def materialize(
        self, observation: Observation, *, episode_id: str
    ) -> TeacherMaterialization:
        """Run real OpenPI preprocessing, prefix prefill, and teacher sampling."""

        if not isinstance(observation, Observation):
            raise TypeError("materialization requires a raw Observation")
        episode_id = _require_nonempty(episode_id, "episode_id")
        self.verified_target.validate_runtime_model()
        self._activate(episode_id)
        model_observation = self.policy.prepare_model_observation(observation)
        bundle = self._verifier.prepare_prefix(model_observation, episode_id=episode_id)
        teacher_output = self.policy.predict_with_features(observation).output
        model_values = teacher_output.action.model_values
        if model_values is None:
            raise ConfigurationError(
                "OpenPI teacher did not preserve pre-output-transform model actions"
            )
        teacher = torch.as_tensor(model_values, device=bundle.device)
        if tuple(teacher.shape) != (
            PI05_ACTION_HORIZON,
            PI05_MODEL_ACTION_DIM,
        ):
            raise ShapeMismatchError("teacher model actions must have shape [50,32]")
        tensors = (
            bundle.prefix_embs,
            bundle.normalized_state,
            teacher,
        )
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ConfigurationError("teacher materialization contains NaN or infinity")
        if bundle.batch_size != 1:
            raise ShapeMismatchError(
                "source materialization requires one raw observation per sample"
            )
        return TeacherMaterialization(
            prefix_embs=bundle.prefix_embs[0].detach(),
            prefix_pad_masks=bundle.prefix_pad_masks[0].detach(),
            prefix_att_masks=bundle.prefix_att_masks[0].detach(),
            robot_state=bundle.normalized_state[0].detach(),
            teacher_actions=teacher.detach(),
        )


def create_teacher_owner_from_env(options: Mapping[str, Any]) -> OpenPITeacherOwner:
    """Create the repository's manifest-bound teacher owner from stack config.

    The actual checkpoint and device inputs remain the source-locked
    ``FIBOCOM_PI05_*`` environment variables consumed by the ordinary policy
    factory. A naked, non-manifest load is rejected even though that factory
    supports a legacy mode for other callers.
    """

    if not isinstance(options, Mapping):
        raise ConfigurationError("teacher owner options must be an object")
    _require_exact_keys(dict(options), {"stack_config"}, name="teacher owner options")
    raw_path = _require_nonempty(options["stack_config"], "stack_config")
    config_path = Path(raw_path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"stack_config is not a file: {config_path}")
    if not os.environ.get("FIBOCOM_PI05_ASSET_MANIFEST", "").strip():
        raise ConfigurationError(
            "FIBOCOM_PI05_ASSET_MANIFEST is required for Draft teacher materialization"
        )
    from ..config import StackConfig
    from ..factories import create_openpi_policy_from_env

    config = StackConfig.from_json(config_path)
    if config.residual_rl.enabled or config.speculative.enabled:
        raise ConfigurationError(
            "Draft teacher materialization requires residual_rl and speculative "
            "to be disabled"
        )
    policy = create_openpi_policy_from_env(config)
    if type(policy) is not RLinfOpenPiChunkPolicy:
        raise ConfigurationError(
            "teacher factory must return the unwrapped RLinfOpenPiChunkPolicy"
        )
    target = getattr(policy, "fibocom_verified_openpi_target", None)
    if not isinstance(target, VerifiedOpenPITarget):
        raise ConfigurationError(
            "teacher policy lacks a manifest-issued VerifiedOpenPITarget"
        )
    return OpenPITeacherOwner(target, policy)


@dataclass(frozen=True)
class VerifiedDraftCache:
    """Cache manifest and shards verified from current local bytes."""

    manifest_path: Path
    manifest_sha256: str
    target_contract: Pi05DraftTargetContract
    source_manifest_sha256: str
    manifest: Mapping[str, Any]
    _proof: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._proof is not _CACHE_PROOF:
            raise ConfigurationError("VerifiedDraftCache must be issued by validation")

    @property
    def sample_count(self) -> int:
        """Return the number of authenticated cache samples."""

        return int(self.manifest["sample_count"])


def _sample_id(episode_id: str, frame_index: int) -> str:
    return _canonical_sha256({"episode_id": episode_id, "frame_index": frame_index})


def _select_episode_splits(
    episode_ids: Iterable[str], *, seed: int, validation_fraction: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    unique = sorted(set(episode_ids))
    if len(unique) < 2:
        raise ConfigurationError(
            "episode-wise train/validation split requires at least two episodes"
        )
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest(),
    )
    validation_count = max(
        1, min(len(ranked) - 1, round(len(ranked) * validation_fraction))
    )
    validation = tuple(sorted(ranked[:validation_count]))
    training = tuple(sorted(ranked[validation_count:]))
    return training, validation


def _as_cache_tensor(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=dtype).contiguous()


def materialize_teacher_cache(
    source: RawObservationSource,
    *,
    source_manifest_path: str | Path,
    teacher_owner: OpenPITeacherOwner,
    output_dir: str | Path,
    split_seed: int = 17,
    validation_fraction: float = 0.2,
) -> VerifiedDraftCache:
    """Materialize an immutable teacher cache from raw observations."""

    if not isinstance(source, RawObservationSource):
        raise TypeError("source must implement RawObservationSource")
    if type(teacher_owner) is not OpenPITeacherOwner:
        raise TypeError("teacher_owner must be an OpenPITeacherOwner")
    if (
        isinstance(split_seed, bool)
        or not isinstance(split_seed, int)
        or split_seed < 0
    ):
        raise ConfigurationError("split_seed must be a non-negative integer")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ConfigurationError("validation_fraction must be strictly between 0 and 1")

    source_path = Path(source_manifest_path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigurationError(f"source manifest is not a file: {source_path}")
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise ConfigurationError(f"cache output directory already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    shards_root = root / "shards"
    shards_root.mkdir()
    copied_source = root / "source_manifest.copy"
    shutil.copyfile(source_path, copied_source)

    seen_lineage: set[tuple[str, int]] = set()
    last_frame_by_episode: dict[str, int] = {}
    previous_actions: dict[str, torch.Tensor] = {}
    materialized: list[dict[str, Any]] = []
    try:
        for ordinal, raw_sample in enumerate(source.iter_samples()):
            if not isinstance(raw_sample, DraftSourceSample):
                raise ConfigurationError(
                    "source adapter yielded a non-DraftSourceSample value; "
                    "model-space cache injection is forbidden"
                )
            lineage = (raw_sample.episode_id, raw_sample.frame_index)
            if lineage in seen_lineage:
                raise ConfigurationError(f"duplicate source lineage: {lineage!r}")
            previous_frame = last_frame_by_episode.get(raw_sample.episode_id)
            if previous_frame is not None and raw_sample.frame_index <= previous_frame:
                raise ConfigurationError(
                    "source frames must be strictly increasing within each episode"
                )
            seen_lineage.add(lineage)
            last_frame_by_episode[raw_sample.episode_id] = raw_sample.frame_index
            owned = teacher_owner.materialize(
                raw_sample.observation, episode_id=raw_sample.episode_id
            )
            last_actions = previous_actions.get(raw_sample.episode_id)
            if last_actions is None:
                last_actions = torch.zeros_like(owned.teacher_actions)
            valid = torch.arange(PI05_ACTION_HORIZON) < raw_sample.valid_steps
            sample_id = _sample_id(*lineage)
            relative = Path("shards") / f"{ordinal:08d}-{sample_id}.safetensors"
            shard_path = root / relative
            tensors = {
                "prefix_embs": _as_cache_tensor(owned.prefix_embs, dtype=torch.float32),
                "prefix_pad_masks": _as_cache_tensor(
                    owned.prefix_pad_masks, dtype=torch.bool
                ),
                "prefix_att_masks": _as_cache_tensor(
                    owned.prefix_att_masks, dtype=torch.bool
                ),
                "robot_state": _as_cache_tensor(owned.robot_state, dtype=torch.float32),
                "last_actions": _as_cache_tensor(last_actions, dtype=torch.float32),
                "teacher_actions": _as_cache_tensor(
                    owned.teacher_actions, dtype=torch.float32
                ),
                "valid_steps": valid.to(dtype=torch.bool),
            }
            save_file(
                tensors,
                str(shard_path),
                metadata={
                    "sample_id": sample_id,
                    "target_contract_sha256": teacher_owner.target_contract.sha256,
                    "production_authorized": "false",
                },
            )
            materialized.append(
                {
                    "sample_id": sample_id,
                    "episode_id": raw_sample.episode_id,
                    "frame_index": raw_sample.frame_index,
                    "path": relative.as_posix(),
                    "size": shard_path.stat().st_size,
                    "sha256": _sha256_file(shard_path),
                    "sample_count": 1,
                }
            )
            previous_actions[raw_sample.episode_id] = owned.teacher_actions.detach()
    finally:
        teacher_owner.close()

    if not materialized:
        raise ConfigurationError("source adapter yielded no samples")
    train_episodes, validation_episodes = _select_episode_splits(
        (value["episode_id"] for value in materialized),
        seed=split_seed,
        validation_fraction=validation_fraction,
    )
    validation_set = set(validation_episodes)
    for sample in materialized:
        sample["split"] = (
            "validation" if sample["episode_id"] in validation_set else "train"
        )
    source_digest = _sha256_file(copied_source)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "production_authorized": False,
        "target_contract": teacher_owner.target_contract.to_dict(),
        "target_contract_sha256": teacher_owner.target_contract.sha256,
        "source_manifest": {
            "path": copied_source.name,
            "size": copied_source.stat().st_size,
            "sha256": source_digest,
        },
        "split": {
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "train_episode_ids": list(train_episodes),
            "validation_episode_ids": list(validation_episodes),
        },
        "sample_count": len(materialized),
        "samples": materialized,
    }
    manifest_path = root / "cache_manifest.json"
    _atomic_json(manifest_path, manifest)
    return verify_teacher_cache(manifest_path)


def _read_cache_tensor_file(
    path: Path,
    *,
    sample_id: str,
    target_sha256: str,
) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = set(handle.keys())
    if keys != _SAMPLE_TENSOR_KEYS:
        raise ConfigurationError(
            f"cache shard tensor keys mismatch for {path.name}: "
            f"missing={sorted(_SAMPLE_TENSOR_KEYS - keys)}, "
            f"unexpected={sorted(keys - _SAMPLE_TENSOR_KEYS)}"
        )
    if metadata != {
        "sample_id": sample_id,
        "target_contract_sha256": target_sha256,
        "production_authorized": "false",
    }:
        raise ConfigurationError(f"cache shard metadata mismatch for {path.name}")
    tensors = load_file(str(path), device="cpu")
    prefix = tensors["prefix_embs"]
    sequence = int(prefix.shape[0]) if prefix.ndim == 2 else -1
    expected_shapes = {
        "prefix_embs": (sequence, PI05_PREFIX_EMBEDDING_DIM),
        "prefix_pad_masks": (sequence,),
        "prefix_att_masks": (sequence,),
        "robot_state": (PI05_MODEL_STATE_DIM,),
        "last_actions": (PI05_ACTION_HORIZON, PI05_MODEL_ACTION_DIM),
        "teacher_actions": (PI05_ACTION_HORIZON, PI05_MODEL_ACTION_DIM),
        "valid_steps": (PI05_ACTION_HORIZON,),
    }
    if sequence <= 0:
        raise ShapeMismatchError("cache prefix must contain at least one token")
    for name, expected in expected_shapes.items():
        if tuple(tensors[name].shape) != expected:
            raise ShapeMismatchError(
                f"cache tensor {name!r} must have shape {expected}"
            )
    for name in ("prefix_pad_masks", "prefix_att_masks", "valid_steps"):
        if tensors[name].dtype != torch.bool:
            raise ConfigurationError(f"cache tensor {name!r} must use bool dtype")
    for name in (
        "prefix_embs",
        "robot_state",
        "last_actions",
        "teacher_actions",
    ):
        if tensors[name].dtype != torch.float32:
            raise ConfigurationError(f"cache tensor {name!r} must use float32 dtype")
        if not bool(torch.isfinite(tensors[name]).all().item()):
            raise ConfigurationError(f"cache tensor {name!r} is non-finite")
    valid = tensors["valid_steps"]
    count = int(valid.to(torch.int64).sum().item())
    if (
        count <= 0
        or not bool(valid[:count].all().item())
        or bool(valid[count:].any().item())
    ):
        raise ConfigurationError("valid_steps must be one non-empty leading prefix")
    if not bool(tensors["prefix_pad_masks"].any().item()):
        raise ConfigurationError("prefix_pad_masks selects no prefix token")
    return tensors


def verify_teacher_cache(manifest_path: str | Path) -> VerifiedDraftCache:
    """Revalidate cache paths, hashes, split isolation, and tensor schemas."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"cache manifest is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"failed to read cache manifest: {error}") from error
    if not isinstance(value, Mapping):
        raise ConfigurationError("cache manifest root must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "production_authorized",
            "target_contract",
            "target_contract_sha256",
            "source_manifest",
            "split",
            "sample_count",
            "samples",
        },
        name="cache manifest",
    )
    if value["schema_version"] != CACHE_SCHEMA_VERSION or value["kind"] != CACHE_KIND:
        raise ConfigurationError("unsupported teacher-cache manifest")
    if value["production_authorized"] is not False:
        raise ConfigurationError("a teacher cache can never authorize production")
    target = Pi05DraftTargetContract.from_dict(value["target_contract"])
    target_sha = _require_sha256(
        value["target_contract_sha256"], "target_contract_sha256"
    )
    if target.sha256 != target_sha:
        raise ConfigurationError("cache target contract digest mismatch")
    root = path.parent
    source = value["source_manifest"]
    if not isinstance(source, Mapping):
        raise ConfigurationError("source_manifest must be an object")
    _require_exact_keys(source, {"path", "size", "sha256"}, name="source_manifest")
    source_path = _safe_child(root, source["path"], "source_manifest.path")
    if not source_path.is_file():
        raise ConfigurationError("cached source manifest is missing")
    if source_path.stat().st_size != source["size"]:
        raise ConfigurationError("cached source manifest size drifted")
    source_sha = _require_sha256(source["sha256"], "source_manifest.sha256")
    if _sha256_file(source_path) != source_sha:
        raise ConfigurationError("cached source manifest SHA-256 drifted")

    split = value["split"]
    if not isinstance(split, Mapping):
        raise ConfigurationError("split must be an object")
    _require_exact_keys(
        split,
        {
            "seed",
            "validation_fraction",
            "train_episode_ids",
            "validation_episode_ids",
        },
        name="split",
    )
    train_episodes = tuple(split["train_episode_ids"])
    validation_episodes = tuple(split["validation_episode_ids"])
    split_seed = split["seed"]
    validation_fraction = split["validation_fraction"]
    if (
        isinstance(split_seed, bool)
        or not isinstance(split_seed, int)
        or split_seed < 0
    ):
        raise ConfigurationError("cache split seed must be non-negative")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or not 0 < float(validation_fraction) < 1
    ):
        raise ConfigurationError("cache validation_fraction must be in (0, 1)")
    if not train_episodes or not validation_episodes:
        raise ConfigurationError(
            "both train and validation episode splits are required"
        )
    if len(set(train_episodes)) != len(train_episodes) or len(
        set(validation_episodes)
    ) != len(validation_episodes):
        raise ConfigurationError("episode split lists contain duplicates")
    if set(train_episodes) & set(validation_episodes):
        raise ConfigurationError("episode-wise train/validation split leaked")

    samples = value["samples"]
    if not isinstance(samples, list) or not samples:
        raise ConfigurationError("cache samples must be a non-empty list")
    if value["sample_count"] != len(samples):
        raise ConfigurationError("cache sample_count mismatch")
    seen_ids: set[str] = set()
    seen_lineage: set[tuple[str, int]] = set()
    seen_paths: set[Path] = set()
    observed_episodes: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ConfigurationError(f"samples[{index}] must be an object")
        _require_exact_keys(
            sample,
            {
                "sample_id",
                "episode_id",
                "frame_index",
                "path",
                "size",
                "sha256",
                "sample_count",
                "split",
            },
            name=f"samples[{index}]",
        )
        episode = _require_nonempty(
            sample["episode_id"], f"samples[{index}].episode_id"
        )
        frame = sample["frame_index"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ConfigurationError("cache frame_index must be non-negative")
        sample_id = _require_sha256(sample["sample_id"], "sample_id")
        if sample_id != _sample_id(episode, frame):
            raise ConfigurationError("cache sample_id does not match lineage")
        lineage = (episode, frame)
        if sample_id in seen_ids or lineage in seen_lineage:
            raise ConfigurationError("cache contains a duplicate sample or lineage")
        seen_ids.add(sample_id)
        seen_lineage.add(lineage)
        observed_episodes.add(episode)
        expected_split = "train" if episode in set(train_episodes) else "validation"
        if episode not in set(train_episodes) | set(validation_episodes):
            raise ConfigurationError("cache sample episode is absent from split lists")
        if sample["split"] != expected_split:
            raise ConfigurationError("cache sample split disagrees with its episode")
        if sample["sample_count"] != 1:
            raise ConfigurationError(
                "this cache schema stores exactly one sample per shard"
            )
        shard_path = _safe_child(root, sample["path"], f"samples[{index}].path")
        if shard_path in seen_paths:
            raise ConfigurationError("cache shard path is duplicated")
        seen_paths.add(shard_path)
        if not shard_path.is_file() or shard_path.stat().st_size != sample["size"]:
            raise ConfigurationError("cache shard is missing or its size drifted")
        shard_sha = _require_sha256(sample["sha256"], "shard sha256")
        if _sha256_file(shard_path) != shard_sha:
            raise ConfigurationError("cache shard SHA-256 drifted")
        _read_cache_tensor_file(
            shard_path, sample_id=sample_id, target_sha256=target_sha
        )
    if observed_episodes != set(train_episodes) | set(validation_episodes):
        raise ConfigurationError("cache split lists contain episodes without samples")
    expected_train, expected_validation = _select_episode_splits(
        observed_episodes,
        seed=split_seed,
        validation_fraction=float(validation_fraction),
    )
    if train_episodes != expected_train or validation_episodes != expected_validation:
        raise ConfigurationError(
            "cache episode split is not the declared deterministic split"
        )
    frozen = json.loads(_canonical_json_bytes(value).decode("utf-8"))
    return VerifiedDraftCache(
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        target_contract=target,
        source_manifest_sha256=source_sha,
        manifest=MappingProxyType(frozen),
        _proof=_CACHE_PROOF,
    )


@dataclass(frozen=True, kw_only=True)
class DraftTrainingConfig:
    """Auditable optimizer and loss settings for one training lineage."""

    total_optimizer_steps: int
    training_code_repository: str
    training_code_revision: str
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    seed: int = 17
    device: str = "cpu"
    loss: Pi05DraftLossConfig = field(default_factory=Pi05DraftLossConfig)

    def __post_init__(self) -> None:
        for name in ("total_optimizer_steps", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ConfigurationError("training seed must be non-negative")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConfigurationError("weight_decay must be finite and non-negative")
        object.__setattr__(self, "device", _require_nonempty(self.device, "device"))
        object.__setattr__(
            self,
            "training_code_repository",
            _require_nonempty(
                self.training_code_repository, "training_code_repository"
            ),
        )
        object.__setattr__(
            self,
            "training_code_revision",
            _require_revision(self.training_code_revision, "training_code_revision"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete canonical training configuration."""

        return {
            "total_optimizer_steps": self.total_optimizer_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "seed": self.seed,
            "device": self.device,
            "loss": {
                "executed_prefix_steps": self.loss.executed_prefix_steps,
                "prefix_weight": self.loss.prefix_weight,
                "tail_weight": self.loss.tail_weight,
                "padding_weight": self.loss.padding_weight,
                "huber_beta": self.loss.huber_beta,
            },
            "training_code_repository": self.training_code_repository,
            "training_code_revision": self.training_code_revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DraftTrainingConfig":
        """Parse an exact training configuration."""

        if not isinstance(value, Mapping):
            raise ConfigurationError("training config must be an object")
        expected = {
            "total_optimizer_steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "seed",
            "device",
            "loss",
            "training_code_repository",
            "training_code_revision",
        }
        _require_exact_keys(value, expected, name="training config")
        loss_value = value["loss"]
        if not isinstance(loss_value, Mapping):
            raise ConfigurationError("training loss config must be an object")
        _require_exact_keys(
            loss_value,
            {
                "executed_prefix_steps",
                "prefix_weight",
                "tail_weight",
                "padding_weight",
                "huber_beta",
            },
            name="training loss config",
        )
        parsed = dict(value)
        parsed["loss"] = Pi05DraftLossConfig(**loss_value)
        return cls(**parsed)

    @property
    def sha256(self) -> str:
        """Return the canonical configuration digest."""

        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class VerifiedDraftTrainingRun:
    """A training run revalidated from all current artifact bytes."""

    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    _proof: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._proof is not _RUN_PROOF:
            raise ConfigurationError(
                "VerifiedDraftTrainingRun must be issued by validation"
            )

    @property
    def optimizer_steps(self) -> int:
        """Return the internally counted completed AdamW steps."""

        return int(self.manifest["optimizer_steps"])

    @property
    def complete(self) -> bool:
        """Return whether the configured step target was reached."""

        return bool(self.manifest["complete"])


def _cache_samples(cache: VerifiedDraftCache, split: str) -> list[Mapping[str, Any]]:
    if split not in {"train", "validation"}:
        raise ConfigurationError("cache split must be train or validation")
    return [value for value in cache.manifest["samples"] if value["split"] == split]


def _load_sample(
    cache: VerifiedDraftCache, sample: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    root = cache.manifest_path.parent
    path = _safe_child(root, sample["path"], "sample.path")
    if _sha256_file(path) != sample["sha256"]:
        raise ConfigurationError("cache shard changed after verification")
    return _read_cache_tensor_file(
        path,
        sample_id=sample["sample_id"],
        target_sha256=cache.target_contract.sha256,
    )


def _collate(
    cache: VerifiedDraftCache,
    samples: list[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    loaded = [_load_sample(cache, sample) for sample in samples]
    max_length = max(int(value["prefix_embs"].shape[0]) for value in loaded)
    batch_size = len(loaded)
    prefix = torch.zeros(
        (batch_size, max_length, PI05_PREFIX_EMBEDDING_DIM), dtype=torch.float32
    )
    pad = torch.zeros((batch_size, max_length), dtype=torch.bool)
    attention = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for index, value in enumerate(loaded):
        length = int(value["prefix_embs"].shape[0])
        prefix[index, :length] = value["prefix_embs"]
        pad[index, :length] = value["prefix_pad_masks"]
        attention[index, :length] = value["prefix_att_masks"]
    return {
        "prefix_embs": prefix.to(device),
        "prefix_pad_masks": pad.to(device),
        "prefix_att_masks": attention.to(device),
        "robot_state": torch.stack([value["robot_state"] for value in loaded]).to(
            device
        ),
        "last_actions": torch.stack([value["last_actions"] for value in loaded]).to(
            device
        ),
        "teacher_actions": torch.stack(
            [value["teacher_actions"] for value in loaded]
        ).to(device),
        "valid_steps": torch.stack([value["valid_steps"] for value in loaded]).to(
            device
        ),
    }


def _forward_loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    config: DraftTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    floating_parameter = next(
        (
            parameter
            for parameter in model.parameters()
            if torch.is_floating_point(parameter)
        ),
        None,
    )
    if floating_parameter is None:
        raise ConfigurationError("Draft model has no floating-point parameter")
    model_dtype = floating_parameter.dtype
    prediction = model(
        prefix_embs=batch["prefix_embs"].to(dtype=model_dtype),
        prefix_pad_masks=batch["prefix_pad_masks"],
        prefix_att_masks=batch["prefix_att_masks"],
        robot_state=batch["robot_state"].to(dtype=model_dtype),
        last_actions=batch["last_actions"].to(dtype=model_dtype),
    )
    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != tuple(
        batch["teacher_actions"].shape
    ):
        raise ShapeMismatchError("Draft prediction must match teacher shape [B,50,32]")
    prediction_for_loss = prediction.to(dtype=torch.float32)
    teacher = batch["teacher_actions"].to(dtype=torch.float32)
    mask = build_pi05_draft_loss_mask(
        batch_size=int(prediction.shape[0]),
        device=prediction.device,
        dtype=torch.float32,
        config=config.loss,
        valid_steps=batch["valid_steps"],
    )
    loss = pi05_draft_distillation_loss(
        prediction_for_loss,
        teacher,
        loss_mask=mask,
        config=config.loss,
    )
    return prediction, loss


def evaluate_draft_model(
    model: nn.Module,
    cache: VerifiedDraftCache,
    config: DraftTrainingConfig,
) -> dict[str, float]:
    """Recompute validation loss and RMS from authenticated held-out samples."""

    cache = verify_teacher_cache(cache.manifest_path)
    samples = _cache_samples(cache, "validation")
    if not samples:
        raise ConfigurationError("validation split contains no samples")
    device = torch.device(config.device)
    was_training = model.training
    model.eval()
    weighted_loss_numerator = 0.0
    weighted_loss_denominator = 0.0
    squared_error = 0.0
    element_count = 0
    with torch.inference_mode():
        for start in range(0, len(samples), config.batch_size):
            batch = _collate(
                cache, samples[start : start + config.batch_size], device=device
            )
            prediction, loss = _forward_loss(model, batch, config)
            loss_mask = build_pi05_draft_loss_mask(
                batch_size=int(prediction.shape[0]),
                device=prediction.device,
                dtype=torch.float32,
                config=config.loss,
                valid_steps=batch["valid_steps"],
            )
            denominator = float(loss_mask.sum().detach().cpu().item())
            weighted_loss_numerator += float(loss.detach().cpu().item()) * denominator
            weighted_loss_denominator += denominator
            valid = batch["valid_steps"][:, :, None].expand_as(prediction)
            error = prediction.to(dtype=torch.float32) - batch["teacher_actions"].to(
                dtype=torch.float32
            )
            selected = error[valid]
            squared_error += float(selected.square().sum().detach().cpu().item())
            element_count += int(selected.numel())
    model.train(was_training)
    if weighted_loss_denominator <= 0 or element_count <= 0:
        raise ConfigurationError("validation produced no metric elements")
    result = {
        "validation_weighted_huber": (
            weighted_loss_numerator / weighted_loss_denominator
        ),
        "validation_rms": math.sqrt(squared_error / element_count),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ConfigurationError("validation metrics are non-finite")
    return result


def _model_lineage(model: nn.Module) -> tuple[str, Mapping[str, Any]]:
    if isinstance(model, Pi05DraftChunkHead):
        return "pi05_draft_chunk_head", model.warm_start_lineage.to_dict()
    return (
        "injected_test_only",
        {
            "kind": "injected_test_only",
            "initial_state_content_sha256": _model_state_content_sha256(model),
        },
    )


def _optimizer_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise ConfigurationError("Draft model has no trainable parameters")
    return parameters


def _restore_rng(state: Mapping[str, Any]) -> None:
    cpu = state.get("cpu_rng_state")
    if not isinstance(cpu, torch.Tensor):
        raise ConfigurationError("run state lacks a CPU RNG tensor")
    torch.set_rng_state(cpu.to(device="cpu"))
    cuda = state.get("cuda_rng_state")
    if cuda is not None:
        if not torch.cuda.is_available():
            raise ConfigurationError(
                "run requires CUDA RNG state but CUDA is unavailable"
            )
        if not isinstance(cuda, list) or not all(
            isinstance(value, torch.Tensor) for value in cuda
        ):
            raise ConfigurationError("CUDA RNG state has an invalid schema")
        torch.cuda.set_rng_state_all(cuda)


def _seed_training(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_run(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cache: VerifiedDraftCache,
    config: DraftTrainingConfig,
    optimizer_steps: int,
    train_loss_sum: float,
    train_loss_count: int,
    train_loss_last: float,
    model_kind: str,
    warm_start: Mapping[str, Any],
    parent_manifest_sha256: str | None,
) -> VerifiedDraftTrainingRun:
    if output_dir.exists():
        raise ConfigurationError(f"run output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.safetensors"
    state_path = output_dir / "training_state.pt"
    _save_model_state(model_path, model)
    validation = evaluate_draft_model(model, cache, config)
    if train_loss_count != optimizer_steps or train_loss_count <= 0:
        raise ConfigurationError("training loss aggregation and step count diverged")
    metrics = {
        "train_weighted_huber_mean": train_loss_sum / train_loss_count,
        "train_weighted_huber_last": train_loss_last,
        **validation,
    }
    state = {
        "optimizer": optimizer.state_dict(),
        "optimizer_steps": optimizer_steps,
        "train_loss_sum": train_loss_sum,
        "train_loss_count": train_loss_count,
        "train_loss_last": train_loss_last,
        "metrics": metrics,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if any(parameter.device.type == "cuda" for parameter in model.parameters())
            else None
        ),
    }
    _atomic_torch_save(state_path, state)
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": RUN_KIND,
        "production_authorized": False,
        "cache_manifest_sha256": cache.manifest_sha256,
        "target_contract": cache.target_contract.to_dict(),
        "target_contract_sha256": cache.target_contract.sha256,
        "training_config": config.to_dict(),
        "training_config_sha256": config.sha256,
        "model_kind": model_kind,
        "warm_start": dict(warm_start),
        "warm_start_sha256": _canonical_sha256(warm_start),
        "model": {
            "path": model_path.name,
            "size": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
            "state_schema": _state_schema(model),
            "state_content_sha256": _model_state_content_sha256(model),
        },
        "training_state": {
            "path": state_path.name,
            "size": state_path.stat().st_size,
            "sha256": _sha256_file(state_path),
        },
        "optimizer_steps": optimizer_steps,
        "complete": optimizer_steps == config.total_optimizer_steps,
        "metrics": metrics,
        "parent_run_manifest_sha256": parent_manifest_sha256,
    }
    manifest_path = output_dir / "run_manifest.json"
    _atomic_json(manifest_path, manifest)
    return verify_training_run(manifest_path)


def train_draft_model(
    cache: VerifiedDraftCache,
    model: nn.Module,
    config: DraftTrainingConfig,
    *,
    output_dir: str | Path,
    resume_run: VerifiedDraftTrainingRun | None = None,
    stop_after_new_steps: int | None = None,
    allow_test_model: bool = False,
) -> VerifiedDraftTrainingRun:
    """Execute real forward/backward/clip/AdamW steps and save a bound run."""

    if not isinstance(cache, VerifiedDraftCache):
        raise TypeError("cache must be a VerifiedDraftCache")
    cache = verify_teacher_cache(cache.manifest_path)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(config, DraftTrainingConfig):
        raise TypeError("config must be a DraftTrainingConfig")
    if stop_after_new_steps is not None and (
        isinstance(stop_after_new_steps, bool)
        or not isinstance(stop_after_new_steps, int)
        or stop_after_new_steps <= 0
    ):
        raise ConfigurationError("stop_after_new_steps must be a positive integer")
    model_kind, warm_start = _model_lineage(model)
    if model_kind == "injected_test_only" and not allow_test_model:
        raise ConfigurationError(
            "only Pi05DraftChunkHead is permitted outside explicit test injection"
        )
    if isinstance(model, Pi05DraftChunkHead):
        revision, dirty = current_git_revision()
        if dirty or revision != config.training_code_revision:
            raise ConfigurationError(
                "pi0.5 Draft training must bind the current clean Git revision"
            )
        if model.production_compatible:
            raise ConfigurationError("production-authorized Draft cannot be trained")
        if model.target_contract != cache.target_contract:
            raise ConfigurationError("Draft head target differs from teacher cache")

    device = torch.device(config.device)
    model.to(device)
    model.train()
    parameters = _optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    completed = 0
    train_loss_sum = 0.0
    train_loss_count = 0
    train_loss_last = 0.0
    parent_sha: str | None = None
    if resume_run is None:
        _seed_training(config.seed)
    else:
        if not isinstance(resume_run, VerifiedDraftTrainingRun):
            raise TypeError("resume_run must be a VerifiedDraftTrainingRun")
        resume_run = verify_training_run(resume_run.manifest_path)
        manifest = resume_run.manifest
        if manifest["cache_manifest_sha256"] != cache.manifest_sha256:
            raise ConfigurationError("resume run is bound to a different cache")
        if manifest["training_config_sha256"] != config.sha256:
            raise ConfigurationError("resume run training config drifted")
        if manifest["target_contract_sha256"] != cache.target_contract.sha256:
            raise ConfigurationError("resume run target contract drifted")
        if manifest["model_kind"] != model_kind:
            raise ConfigurationError("resume run model kind drifted")
        if manifest["warm_start_sha256"] != _canonical_sha256(warm_start):
            raise ConfigurationError("resume run warm-start lineage drifted")
        model_path = _safe_child(
            resume_run.manifest_path.parent,
            manifest["model"]["path"],
            "model.path",
        )
        _load_model_state(model_path, model, device)
        state_path = _safe_child(
            resume_run.manifest_path.parent,
            manifest["training_state"]["path"],
            "training_state.path",
        )
        state = _safe_torch_load(state_path, map_location=device)
        if not isinstance(state, Mapping) or set(state) != {
            "optimizer",
            "optimizer_steps",
            "train_loss_sum",
            "train_loss_count",
            "train_loss_last",
            "metrics",
            "cpu_rng_state",
            "cuda_rng_state",
        }:
            raise ConfigurationError("training state schema mismatch")
        completed = state["optimizer_steps"]
        train_loss_sum = float(state["train_loss_sum"])
        train_loss_count = state["train_loss_count"]
        train_loss_last = float(state["train_loss_last"])
        if completed != resume_run.optimizer_steps:
            raise ConfigurationError("resume optimizer step count drifted")
        if train_loss_count != completed:
            raise ConfigurationError("resume training loss aggregation drifted")
        optimizer.load_state_dict(state["optimizer"])
        _restore_rng(state)
        parent_sha = resume_run.manifest_sha256
        if isinstance(model, Pi05DraftChunkHead):
            model._optimizer_steps = completed  # noqa: SLF001

    if completed >= config.total_optimizer_steps:
        raise ConfigurationError(
            "resume run already reached its configured step target"
        )
    samples = _cache_samples(cache, "train")
    if not samples:
        raise ConfigurationError("training split contains no samples")
    new_limit = config.total_optimizer_steps - completed
    if stop_after_new_steps is not None:
        new_limit = min(new_limit, stop_after_new_steps)
    for _ in range(new_limit):
        start = (completed * config.batch_size) % len(samples)
        selected = [
            samples[(start + offset) % len(samples)]
            for offset in range(config.batch_size)
        ]
        batch = _collate(cache, selected, device=device)
        optimizer.zero_grad(set_to_none=True)
        _prediction, loss = _forward_loss(model, batch, config)
        if loss.ndim != 0 or not loss.requires_grad or loss.grad_fn is None:
            raise ConfigurationError(
                "internally computed Draft loss must be a differentiable scalar"
            )
        if not bool(torch.isfinite(loss.detach()).item()):
            raise ConfigurationError("Draft loss is non-finite; optimizer step aborted")
        loss.backward()
        gradients = [value.grad for value in parameters if value.grad is not None]
        if not gradients:
            raise ConfigurationError("Draft backward pass produced no gradients")
        if not all(bool(torch.isfinite(value).all().item()) for value in gradients):
            optimizer.zero_grad(set_to_none=True)
            raise ConfigurationError(
                "Draft gradients are non-finite; optimizer step aborted"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, config.gradient_clip_norm, error_if_nonfinite=True
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            optimizer.zero_grad(set_to_none=True)
            raise ConfigurationError("Draft clipped gradient norm is non-finite")
        optimizer.step()
        if not all(bool(torch.isfinite(value).all().item()) for value in parameters):
            raise ConfigurationError("optimizer produced non-finite Draft parameters")
        completed += 1
        train_loss_last = float(loss.detach().cpu().item())
        train_loss_sum += train_loss_last
        train_loss_count += 1
        if isinstance(model, Pi05DraftChunkHead):
            model.record_completed_optimizer_step(loss)
            if model.training_step_count != completed:
                raise ConfigurationError(
                    "Draft head and trainer step counters diverged"
                )
    return _save_run(
        output_dir=Path(output_dir).expanduser().resolve(),
        model=model,
        optimizer=optimizer,
        cache=cache,
        config=config,
        optimizer_steps=completed,
        train_loss_sum=train_loss_sum,
        train_loss_count=train_loss_count,
        train_loss_last=train_loss_last,
        model_kind=model_kind,
        warm_start=warm_start,
        parent_manifest_sha256=parent_sha,
    )


def verify_training_run(manifest_path: str | Path) -> VerifiedDraftTrainingRun:
    """Verify every run artifact hash and all lineage relationships."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"run manifest is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"failed to read run manifest: {error}") from error
    if not isinstance(value, Mapping):
        raise ConfigurationError("run manifest root must be an object")
    expected = {
        "schema_version",
        "kind",
        "production_authorized",
        "cache_manifest_sha256",
        "target_contract",
        "target_contract_sha256",
        "training_config",
        "training_config_sha256",
        "model_kind",
        "warm_start",
        "warm_start_sha256",
        "model",
        "training_state",
        "optimizer_steps",
        "complete",
        "metrics",
        "parent_run_manifest_sha256",
    }
    _require_exact_keys(value, expected, name="run manifest")
    if value["schema_version"] != RUN_SCHEMA_VERSION or value["kind"] != RUN_KIND:
        raise ConfigurationError("unsupported training-run manifest")
    if value["production_authorized"] is not False:
        raise ConfigurationError("a training run can never authorize production")
    cache_sha = _require_sha256(value["cache_manifest_sha256"], "cache manifest sha256")
    del cache_sha
    target = Pi05DraftTargetContract.from_dict(value["target_contract"])
    if target.sha256 != _require_sha256(
        value["target_contract_sha256"], "target contract sha256"
    ):
        raise ConfigurationError("run target contract digest mismatch")
    config = DraftTrainingConfig.from_dict(value["training_config"])
    if config.sha256 != _require_sha256(
        value["training_config_sha256"], "training config sha256"
    ):
        raise ConfigurationError("run training config digest mismatch")
    if _canonical_sha256(value["warm_start"]) != _require_sha256(
        value["warm_start_sha256"], "warm start sha256"
    ):
        raise ConfigurationError("run warm-start lineage digest mismatch")
    if value["model_kind"] not in {"pi05_draft_chunk_head", "injected_test_only"}:
        raise ConfigurationError("unsupported run model kind")
    if value["model_kind"] == "pi05_draft_chunk_head":
        WarmStartLineage.from_dict(value["warm_start"])
    root = path.parent
    for name in ("model", "training_state"):
        artifact = value[name]
        if not isinstance(artifact, Mapping):
            raise ConfigurationError(f"run {name} must be an object")
        required = (
            {"path", "size", "sha256", "state_schema", "state_content_sha256"}
            if name == "model"
            else {"path", "size", "sha256"}
        )
        _require_exact_keys(artifact, required, name=f"run {name}")
        artifact_path = _safe_child(root, artifact["path"], f"{name}.path")
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != artifact["size"]
        ):
            raise ConfigurationError(f"run {name} is missing or size drifted")
        if _sha256_file(artifact_path) != _require_sha256(
            artifact["sha256"], f"{name}.sha256"
        ):
            raise ConfigurationError(f"run {name} SHA-256 drifted")
    model_path = _safe_child(root, value["model"]["path"], "model.path")
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        state_schema = {
            name: {
                "shape": list(handle.get_tensor(name).shape),
                "dtype": str(handle.get_tensor(name).dtype).removeprefix("torch."),
            }
            for name in sorted(handle.keys())
        }
    if state_schema != value["model"]["state_schema"]:
        raise ConfigurationError("run model state schema drifted")
    if _state_file_content_sha256(model_path) != _require_sha256(
        value["model"]["state_content_sha256"], "model.state_content_sha256"
    ):
        raise ConfigurationError("run model state content digest drifted")
    steps = value["optimizer_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ConfigurationError("run optimizer_steps must be positive")
    expected_complete = steps == config.total_optimizer_steps
    if (
        steps > config.total_optimizer_steps
        or value["complete"] is not expected_complete
    ):
        raise ConfigurationError(
            "run completion flag or optimizer step count is invalid"
        )
    metrics = value["metrics"]
    expected_metric_keys = {
        "train_weighted_huber_mean",
        "train_weighted_huber_last",
        "validation_weighted_huber",
        "validation_rms",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_metric_keys:
        raise ConfigurationError("run metrics schema mismatch")
    if not all(
        isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
        for metric in metrics.values()
    ):
        raise ConfigurationError("run metrics are non-finite")
    state_path = _safe_child(
        root, value["training_state"]["path"], "training_state.path"
    )
    state = _safe_torch_load(state_path, map_location="cpu")
    if not isinstance(state, Mapping) or set(state) != {
        "optimizer",
        "optimizer_steps",
        "train_loss_sum",
        "train_loss_count",
        "train_loss_last",
        "metrics",
        "cpu_rng_state",
        "cuda_rng_state",
    }:
        raise ConfigurationError("run training state schema mismatch")
    aggregate_values = (
        state["train_loss_sum"],
        state["train_loss_last"],
    )
    if not all(
        isinstance(number, (int, float))
        and not isinstance(number, bool)
        and math.isfinite(float(number))
        for number in aggregate_values
    ):
        raise ConfigurationError("run training loss aggregation is non-finite")
    if state["train_loss_count"] != steps:
        raise ConfigurationError("run training loss count differs from optimizer steps")
    expected_mean = float(state["train_loss_sum"]) / steps
    if (
        state["optimizer_steps"] != steps
        or state["metrics"] != metrics
        or not math.isclose(
            expected_mean,
            float(metrics["train_weighted_huber_mean"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(state["train_loss_last"]),
            float(metrics["train_weighted_huber_last"]),
            rel_tol=0,
            abs_tol=0,
        )
    ):
        raise ConfigurationError(
            "run manifest counters or metrics differ from trainer-owned state"
        )
    parent = value["parent_run_manifest_sha256"]
    if parent is not None:
        _require_sha256(parent, "parent_run_manifest_sha256")
    frozen = json.loads(_canonical_json_bytes(value).decode("utf-8"))
    return VerifiedDraftTrainingRun(
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        manifest=MappingProxyType(frozen),
        _proof=_RUN_PROOF,
    )


def load_run_weights(
    run: VerifiedDraftTrainingRun,
    model: nn.Module,
    *,
    device: str | torch.device,
) -> None:
    """Strictly load one verified run's model bytes into a matching module."""

    run = verify_training_run(run.manifest_path)
    model_path = _safe_child(
        run.manifest_path.parent, run.manifest["model"]["path"], "model.path"
    )
    _load_model_state(model_path, model, torch.device(device))
    if (
        _model_state_content_sha256(model)
        != run.manifest["model"]["state_content_sha256"]
    ):
        raise ConfigurationError("loaded run model content digest mismatch")


def export_candidate_from_run(
    run: VerifiedDraftTrainingRun,
    cache: VerifiedDraftCache,
    head: Pi05DraftChunkHead,
    *,
    output_dir: str | Path,
) -> Any:
    """Recompute held-out metrics and export an unsigned candidate only."""

    if not isinstance(run, VerifiedDraftTrainingRun):
        raise TypeError("run must be a VerifiedDraftTrainingRun")
    if not isinstance(cache, VerifiedDraftCache):
        raise TypeError("cache must be a VerifiedDraftCache")
    if type(head) is not Pi05DraftChunkHead:
        raise TypeError("candidate export requires an exact Pi05DraftChunkHead")
    run = verify_training_run(run.manifest_path)
    cache = verify_teacher_cache(cache.manifest_path)
    manifest = run.manifest
    if not run.complete:
        raise ConfigurationError("incomplete training run cannot be exported")
    if manifest["model_kind"] != "pi05_draft_chunk_head":
        raise ConfigurationError("test-injected training runs cannot be exported")
    if manifest["cache_manifest_sha256"] != cache.manifest_sha256:
        raise ConfigurationError("run and cache manifest digests differ")
    if manifest["target_contract_sha256"] != head.target_contract.sha256:
        raise ConfigurationError("run and export head target contracts differ")
    if manifest["warm_start_sha256"] != _canonical_sha256(
        head.warm_start_lineage.to_dict()
    ):
        raise ConfigurationError("run and export head warm-start lineage differ")
    config = DraftTrainingConfig.from_dict(manifest["training_config"])
    head.to(torch.device(config.device))
    load_run_weights(run, head, device=config.device)
    head._optimizer_steps = run.optimizer_steps  # noqa: SLF001
    recomputed = evaluate_draft_model(head, cache, config)
    recorded = manifest["metrics"]
    for name in ("validation_weighted_huber", "validation_rms"):
        if not math.isclose(
            recomputed[name], float(recorded[name]), rel_tol=1e-6, abs_tol=1e-7
        ):
            raise ConfigurationError(
                f"recorded validation metric {name} does not match recomputation"
            )
    training_record = Pi05DraftTrainingRecord(
        dataset_manifest_sha256=cache.manifest_sha256,
        training_code_repository=config.training_code_repository,
        training_code_revision=config.training_code_revision,
        hparams=config.to_dict(),
        seed=config.seed,
        optimizer_steps=run.optimizer_steps,
        metrics={
            "train_weighted_huber_mean": float(recorded["train_weighted_huber_mean"]),
            "held_out_weighted_huber": recomputed["validation_weighted_huber"],
            "held_out_rms": recomputed["validation_rms"],
        },
    )
    artifact = export_pi05_draft_candidate(
        head,
        training_record=training_record,
        output_dir=output_dir,
    )
    manifest_value = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    if manifest_value.get("production_authorized") is not False:
        raise ConfigurationError(
            "candidate exporter unexpectedly authorized production"
        )
    return artifact


def current_git_revision() -> tuple[str, bool]:
    """Return the current commit and dirty flag without inventing a revision."""

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
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigurationError(
            "training requires an identifiable Git checkout"
        ) from error
    return _require_revision(revision, "git revision"), dirty


__all__ = [
    "CACHE_KIND",
    "CACHE_SCHEMA_VERSION",
    "RUN_KIND",
    "RUN_SCHEMA_VERSION",
    "DraftSourceSample",
    "DraftTrainingConfig",
    "OpenPITeacherOwner",
    "RawObservationSource",
    "TeacherMaterialization",
    "VerifiedDraftCache",
    "VerifiedDraftTrainingRun",
    "create_teacher_owner_from_env",
    "current_git_revision",
    "evaluate_draft_model",
    "export_candidate_from_run",
    "load_run_weights",
    "load_source_adapter",
    "materialize_teacher_cache",
    "train_draft_model",
    "verify_teacher_cache",
    "verify_training_run",
]
