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

"""Checksummed residual-training checkpoints with explicit lineage."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

CHECKPOINT_FORMAT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_residual_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically save a task-owned checkpoint and its SHA-256 sidecar."""

    destination = Path(path)
    if destination.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("residual checkpoint must use .pt or .pth")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint payload has the wrong format_version")
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii"
    )
    return digest


def load_residual_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    require_sidecar: bool = True,
) -> tuple[dict[str, Any], str]:
    """Verify and safely load a tensor-only checkpoint.

    ``weights_only=True`` deliberately rejects arbitrary pickle globals. A
    checkpoint from an older or unrelated tool must be converted explicitly,
    rather than weakening this loading boundary.
    """

    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if require_sidecar and not sidecar.is_file():
        raise FileNotFoundError(f"missing checkpoint checksum sidecar: {sidecar}")
    observed_digest = file_sha256(source)
    if sidecar.is_file():
        expected_digest = sidecar.read_text(encoding="ascii").split()[0]
        if expected_digest != observed_digest:
            raise ValueError("checkpoint SHA-256 does not match its sidecar")
    try:
        payload = torch.load(
            source,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "this PyTorch version lacks safe weights_only checkpoint loading"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("residual checkpoint payload must be a mapping")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')}"
        )
    if not isinstance(payload.get("trainer_state"), dict):
        raise ValueError("checkpoint is missing trainer_state")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("checkpoint is missing metadata")
    return payload, observed_digest
