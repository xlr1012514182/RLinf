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
from pathlib import Path
from typing import Any

import pytest
import torch

from rlinf.projects.fibocom_vla import draft_assets
from rlinf.projects.fibocom_vla.draft_assets import (
    DraftAsset,
    DraftCompatibility,
    DraftResolution,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError
from rlinf.projects.fibocom_vla.inference import draft_head
from rlinf.projects.fibocom_vla.inference.draft_head import (
    BASE_MODEL_FAMILY,
    PINNED_TRAIN_SCRIPT_ARCH,
    PUBLISHED_DRAFT_ARCH,
    PUBLISHED_SAMPLE_SEMANTICS,
    SOURCE_DIVERGENCES,
    SOURCE_REVISION,
    SOURCE_SHA256,
    DraftChunkHead,
    load_draft_chunk_head,
)


def _published_meta() -> dict[str, Any]:
    return {
        "chunk_m": 50,
        "out_dim": 7,
        "img_dim": 2048,
        "draft_arch": "vlm1_query_decoder",
        "draft_input_mode": "prefix_embs",
        "draft_hidden_size": 2048,
        "draft_num_heads": 8,
        "draft_num_kv_heads": 1,
        "draft_head_dim": 256,
        "use_last_actions": False,
        "use_seed_actions": False,
        "sample_semantics": "sliding_chunk_shift_v2",
        # Audited training fields are allowed but cannot weaken required meta.
        "target_source": "teacher_zero_noise",
        "teacher_noise_mode": "zero",
    }


def _published_meta_state_dict() -> dict[str, torch.Tensor]:
    # Meta tensors preserve exact production geometry/dtypes in a tiny test
    # checkpoint instead of allocating the 441 MB published state dict.
    return {
        name: torch.empty(
            shape, device="meta", dtype=draft_head._EXPECTED_STATE_DTYPES[name]
        )
        for name, shape in draft_head._EXPECTED_STATE_SHAPES.items()
    }


def _write_resolution(
    tmp_path: Path,
    *,
    production_compatible: bool,
    checkpoint: dict[str, Any] | None = None,
) -> DraftResolution:
    path = tmp_path / "draft_test.pt"
    torch.save(
        checkpoint
        or {
            "draft_head": _published_meta_state_dict(),
            "meta": _published_meta(),
        },
        path,
    )
    payload = path.read_bytes()
    verified = DraftAsset(
        suite="libero_goal",
        filename=Path(path.name),
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    ).verify(tmp_path)
    compatibility = DraftCompatibility(
        purpose=(
            "production" if production_compatible else "warm_start_initialization_only"
        ),
        production_compatible=production_compatible,
        initialization_allowed=not production_compatible,
        mismatches=() if production_compatible else ("model_family: pi0 -> pi05",),
    )
    return DraftResolution(
        verified_asset=verified,
        compatibility=compatibility,
    )


def test_source_revision_sha_and_published_label_divergence_are_explicit() -> None:
    assert SOURCE_REVISION == draft_assets.PUBLISHED_CODE_REVISION
    assert SOURCE_SHA256 == (
        "186deb7f6cf89c3476b1b5f6d37350dc7a6073c55367d4edb99c89dea8c3793d"
    )
    assert PUBLISHED_DRAFT_ARCH == "vlm1_query_decoder"
    assert PINNED_TRAIN_SCRIPT_ARCH == "vlm_block"
    assert PUBLISHED_SAMPLE_SEMANTICS == "sliding_chunk_shift_v2"
    assert any("vlm1_query_decoder" in value for value in SOURCE_DIVERGENCES)
    assert any("vlm_block" in value for value in SOURCE_DIVERGENCES)


def test_equivalent_small_query_decoder_runs_and_preserves_tensor_contract() -> None:
    head = DraftChunkHead(
        img_dim=16,
        chunk_m=2,
        hidden_dim=32,
        out_dim=3,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )

    output = head(
        prefix_embs=torch.randn(1, 4, 16),
        prefix_pad_masks=torch.ones(1, 4, dtype=torch.bool),
        prefix_att_masks=torch.zeros(1, 4, dtype=torch.bool),
        robot_state=torch.randn(1, 32),
        last_actions=torch.randn(1, 2, 3),
    )

    assert output.shape == (1, 2, 3)
    assert output.dtype == torch.float32
    assert head.initialization_only is True
    assert head.production_compatible is False
    assert head.base_model_family == BASE_MODEL_FAMILY == "pi0"
    assert head.pi05_compatible is False


def test_incompatible_resolution_only_allows_warm_start_and_blocks_adapter(
    tmp_path: Path,
) -> None:
    resolution = _write_resolution(tmp_path, production_compatible=False)

    with pytest.raises(ConfigurationError, match="does not authorize production"):
        load_draft_chunk_head(
            resolution,
            purpose="production",
            device="meta",
        )

    head = load_draft_chunk_head(
        resolution,
        purpose="warm_start",
        device="meta",
    )

    assert head.initialization_only is True
    assert head.production_compatible is False
    assert head.base_model_family == "pi0"
    assert head.checkpoint_metadata["draft_arch"] == PUBLISHED_DRAFT_ARCH
    assert head.training is True
    assert all(parameter.requires_grad for parameter in head.parameters())
    with pytest.raises(ConfigurationError, match="initialization-only"):
        head.production_predict_adapter()


def test_production_resolution_strict_load_enables_guarded_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolution = _write_resolution(tmp_path, production_compatible=True)
    original_load_state_dict = DraftChunkHead.load_state_dict
    observed: dict[str, object] = {}

    def recording_load(
        self: DraftChunkHead,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        observed["strict"] = strict
        observed["assign"] = assign
        return original_load_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )

    monkeypatch.setattr(DraftChunkHead, "load_state_dict", recording_load)
    head = load_draft_chunk_head(
        resolution,
        purpose="production",
        device="meta",
    )

    assert observed == {"strict": True, "assign": False}
    assert head.initialization_only is False
    assert head.production_compatible is True
    assert head.base_model_family == "pi0"
    assert head.pi05_compatible is False
    assert head.training is False
    assert all(not parameter.requires_grad for parameter in head.parameters())
    adapter = head.production_predict_adapter()
    assert adapter.head is head

    def recording_forward(**inputs: torch.Tensor) -> torch.Tensor:
        observed["inference_mode"] = torch.is_inference_mode_enabled()
        return inputs["value"]

    monkeypatch.setattr(head, "forward", recording_forward)
    value = torch.ones(1)
    assert adapter.predict(value=value) is value
    assert observed["inference_mode"] is True

    head.train()
    with pytest.raises(ConfigurationError, match="remain in eval mode"):
        adapter.predict(value=value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("chunk_m", 49),
        ("out_dim", 14),
        ("img_dim", 1024),
        ("draft_arch", "vlm_block"),
        ("draft_input_mode", "last_actions"),
        ("draft_num_heads", 4),
        ("draft_num_kv_heads", 2),
        ("draft_head_dim", 128),
        ("sample_semantics", "sliding_chunk_shift_v1"),
        ("model_family", "pi05"),
    ],
)
def test_loader_rejects_checkpoint_meta_drift(
    tmp_path: Path, field: str, replacement: object
) -> None:
    meta = _published_meta()
    meta[field] = replacement
    resolution = _write_resolution(
        tmp_path,
        production_compatible=True,
        checkpoint={
            "draft_head": _published_meta_state_dict(),
            "meta": meta,
        },
    )

    with pytest.raises(ConfigurationError, match=field):
        load_draft_chunk_head(
            resolution,
            purpose="production",
            device="meta",
        )


@pytest.mark.parametrize("mutation", ["missing", "shape", "dtype", "unexpected"])
def test_loader_rejects_state_dict_schema_shape_and_dtype_drift(
    tmp_path: Path, mutation: str
) -> None:
    state_dict = _published_meta_state_dict()
    if mutation == "missing":
        state_dict.pop("_action_head.bias")
    elif mutation == "shape":
        state_dict["_action_head.bias"] = torch.empty(8, device="meta")
    elif mutation == "dtype":
        state_dict["_action_head.bias"] = torch.empty(
            7, device="meta", dtype=torch.float16
        )
    else:
        state_dict["unexpected.weight"] = torch.empty(1, device="meta")
    resolution = _write_resolution(
        tmp_path,
        production_compatible=True,
        checkpoint={"draft_head": state_dict, "meta": _published_meta()},
    )

    with pytest.raises(ConfigurationError, match="state_dict|shape|dtype"):
        load_draft_chunk_head(
            resolution,
            purpose="production",
            device="meta",
        )


def test_loader_requires_draft_resolution_instead_of_loose_path(tmp_path: Path) -> None:
    path = tmp_path / "loose.pt"
    path.write_bytes(b"not accepted")

    with pytest.raises(TypeError, match="DraftResolution"):
        load_draft_chunk_head(  # type: ignore[arg-type]
            path,
            purpose="production",
            device="meta",
        )
