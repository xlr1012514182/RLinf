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

"""Pickle-free, checksummed archives for residual-policy rollouts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..errors import ConfigurationError, ShapeMismatchError
from .trajectory import ActionSource, ResidualRollout

FORMAT_VERSION = 1
ROLLOUT_FIELDS = (
    "features",
    "next_features",
    "reference_actions",
    "next_reference_actions",
    "residual_actions",
    "old_log_probs",
    "action_source",
    "policy_mask",
    "rewards",
    "terminals",
    "valid_mask",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    descriptor = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": contiguous.shape},
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(descriptor + memoryview(contiguous).cast("B"))


@dataclass(frozen=True)
class RolloutArchiveMetadata:
    """Identity and collection context stored beside rollout tensors."""

    task: str
    base_model: str
    base_model_revision: str
    behavior_policy_revision: str
    seed: int
    configuration_sha256: str
    collected_at_utc: str = ""
    notes: str = ""

    def normalized(self) -> "RolloutArchiveMetadata":
        """Fill the timestamp while preserving all caller-supplied lineage."""

        timestamp = self.collected_at_utc or datetime.now(timezone.utc).isoformat()
        return RolloutArchiveMetadata(
            task=self.task,
            base_model=self.base_model,
            base_model_revision=self.base_model_revision,
            behavior_policy_revision=self.behavior_policy_revision,
            seed=self.seed,
            configuration_sha256=self.configuration_sha256,
            collected_at_utc=timestamp,
            notes=self.notes,
        )

    def validate(self) -> None:
        """Reject archives without enough identity to reproduce collection."""

        required = {
            "task": self.task,
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "behavior_policy_revision": self.behavior_policy_revision,
            "configuration_sha256": self.configuration_sha256,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ConfigurationError(
                f"rollout metadata fields must not be empty: {', '.join(missing)}"
            )
        if self.seed < 0:
            raise ConfigurationError("rollout seed must be non-negative")


@dataclass(frozen=True)
class ResidualEpisode:
    """One unpadded, terminal chunk trajectory ready for archival."""

    features: NDArray[np.float32]
    next_features: NDArray[np.float32]
    reference_actions: NDArray[np.float32]
    next_reference_actions: NDArray[np.float32]
    residual_actions: NDArray[np.float32]
    old_log_probs: NDArray[np.float32]
    action_source: NDArray[np.int8]
    policy_mask: NDArray[np.bool_]
    rewards: NDArray[np.float32]
    terminals: NDArray[np.bool_]

    def __post_init__(self) -> None:
        length = np.asarray(self.rewards).shape
        if len(length) != 1 or length[0] == 0:
            raise ShapeMismatchError("episode rewards must be non-empty [T]")
        valid = np.ones((1, length[0]), dtype=bool)
        rollout = ResidualRollout(
            features=np.asarray(self.features)[None, ...],
            next_features=np.asarray(self.next_features)[None, ...],
            reference_actions=np.asarray(self.reference_actions)[None, ...],
            next_reference_actions=np.asarray(self.next_reference_actions)[None, ...],
            residual_actions=np.asarray(self.residual_actions)[None, ...],
            old_log_probs=np.asarray(self.old_log_probs)[None, ...],
            action_source=np.asarray(self.action_source)[None, ...],
            policy_mask=np.asarray(self.policy_mask)[None, ...],
            rewards=np.asarray(self.rewards)[None, ...],
            terminals=np.asarray(self.terminals)[None, ...],
            valid_mask=valid,
        )
        if not rollout.terminals[0, -1]:
            raise ShapeMismatchError("a complete episode must end with terminal=True")
        if np.any(rollout.terminals[0, :-1]):
            raise ShapeMismatchError("terminal=True may only appear at episode end")
        for name in ROLLOUT_FIELDS:
            if name == "valid_mask":
                continue
            object.__setattr__(self, name, getattr(rollout, name)[0])

    @property
    def length(self) -> int:
        """Number of valid chunk transitions."""

        return int(self.rewards.shape[0])


def pad_residual_episodes(episodes: Sequence[ResidualEpisode]) -> ResidualRollout:
    """Pad complete episodes without inventing policy likelihoods."""

    if not episodes:
        raise ValueError("at least one residual episode is required")
    maximum_length = max(episode.length for episode in episodes)
    batch_size = len(episodes)
    feature_dim = episodes[0].features.shape[-1]
    action_tail = episodes[0].reference_actions.shape[-2:]
    features = np.zeros((batch_size, maximum_length, feature_dim), dtype=np.float32)
    next_features = np.zeros_like(features)
    action_shape = (batch_size, maximum_length, *action_tail)
    reference_actions = np.zeros(action_shape, dtype=np.float32)
    next_reference_actions = np.zeros_like(reference_actions)
    residual_actions = np.zeros_like(reference_actions)
    old_log_probs = np.full((batch_size, maximum_length), np.nan, dtype=np.float32)
    action_source = np.full(
        (batch_size, maximum_length), ActionSource.PADDING, dtype=np.int8
    )
    policy_mask = np.zeros((batch_size, maximum_length), dtype=bool)
    rewards = np.zeros((batch_size, maximum_length), dtype=np.float32)
    terminals = np.zeros((batch_size, maximum_length), dtype=bool)
    valid_mask = np.zeros((batch_size, maximum_length), dtype=bool)

    for index, episode in enumerate(episodes):
        if episode.features.shape[-1] != feature_dim:
            raise ShapeMismatchError("all episodes must share feature_dim")
        if episode.reference_actions.shape[-2:] != action_tail:
            raise ShapeMismatchError("all episodes must share action geometry")
        length = episode.length
        for name in ROLLOUT_FIELDS:
            if name == "valid_mask":
                continue
            destination = locals()[name]
            destination[index, :length] = getattr(episode, name)
        valid_mask[index, :length] = True

    return ResidualRollout(
        features=features,
        next_features=next_features,
        reference_actions=reference_actions,
        next_reference_actions=next_reference_actions,
        residual_actions=residual_actions,
        old_log_probs=old_log_probs,
        action_source=action_source,
        policy_mask=policy_mask,
        rewards=rewards,
        terminals=terminals,
        valid_mask=valid_mask,
    )


def _rollout_arrays(rollout: ResidualRollout) -> dict[str, NDArray[Any]]:
    return {name: np.asarray(getattr(rollout, name)) for name in ROLLOUT_FIELDS}


def save_rollout_archive(
    path: str | Path,
    rollout: ResidualRollout,
    metadata: RolloutArchiveMetadata,
) -> str:
    """Atomically save an NPZ plus SHA-256 sidecar and return its digest."""

    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("rollout archive must use the .npz extension")
    normalized_metadata = metadata.normalized()
    normalized_metadata.validate()
    arrays = _rollout_arrays(rollout)
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "metadata": asdict(normalized_metadata),
        "summary": {
            "episodes": rollout.batch_size,
            "valid_transitions": rollout.valid_transitions,
            "policy_transitions": rollout.policy_transitions,
        },
        "arrays": {
            name: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": _array_digest(array),
            }
            for name, array in arrays.items()
        },
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                manifest_json=np.frombuffer(manifest_bytes, dtype=np.uint8),
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    digest = _sha256_file(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    return digest


def load_rollout_archive(
    path: str | Path, *, require_sidecar: bool = True
) -> tuple[ResidualRollout, RolloutArchiveMetadata, Mapping[str, Any]]:
    """Load a trusted numeric archive after file and per-array verification."""

    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if require_sidecar and not sidecar.is_file():
        raise FileNotFoundError(f"missing rollout checksum sidecar: {sidecar}")
    if sidecar.is_file():
        expected_digest = sidecar.read_text(encoding="ascii").split()[0]
        observed_digest = _sha256_file(source)
        if observed_digest != expected_digest:
            raise ValueError("rollout archive SHA-256 does not match its sidecar")

    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(set(ROLLOUT_FIELDS) - set(archive.files))
        if missing or "manifest_json" not in archive.files:
            raise ValueError(f"rollout archive is missing fields: {missing}")
        manifest = json.loads(archive["manifest_json"].tobytes().decode("utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported rollout format version: {manifest.get('format_version')}"
            )
        arrays = {name: np.asarray(archive[name]) for name in ROLLOUT_FIELDS}

    manifest_arrays = manifest.get("arrays", {})
    for name, array in arrays.items():
        expected = manifest_arrays.get(name, {})
        if expected.get("sha256") != _array_digest(array):
            raise ValueError(f"rollout array digest mismatch: {name}")
        if expected.get("dtype") != array.dtype.str or expected.get("shape") != list(
            array.shape
        ):
            raise ValueError(f"rollout array schema mismatch: {name}")
    rollout = ResidualRollout(**arrays)
    metadata = RolloutArchiveMetadata(**manifest["metadata"])
    metadata.validate()
    summary = manifest.get("summary", {})
    observed_summary = {
        "episodes": rollout.batch_size,
        "valid_transitions": rollout.valid_transitions,
        "policy_transitions": rollout.policy_transitions,
    }
    if summary != observed_summary:
        raise ValueError("rollout manifest summary does not match its tensors")
    return rollout, metadata, manifest
