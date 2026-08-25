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

"""Source-locked Realtime-VLA FLASH Draft head and checkpoint loader.

The query decoder is adapted from Dexmal's Apache-2.0 implementation at
``src/openpi/models_pytorch/draft.py``.  The source identity is recorded by
``SOURCE_REVISION`` and ``SOURCE_SHA256`` below.

There is one material publication/source-label difference which must remain
visible: the published checkpoints call this architecture
``vlm1_query_decoder``, while the training helper at the pinned code revision
emits ``vlm_block``.  The tensors and the referenced ``DraftChunkHead``
implementation agree; this loader accepts only the label found in the
published, SHA-256-pinned checkpoints.  It does not reinterpret those pi0
LIBERO weights as pi0.5 weights.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    GemmaDecoderLayer,
    GemmaRotaryEmbedding,
)

from ..draft_assets import DraftResolution
from ..errors import ConfigurationError

SOURCE_REPOSITORY = "https://github.com/dexmal/realtime-vla-flash"
SOURCE_REVISION = "da6ceccad603695a8a3d6fa14dd410c3aadb536f"
SOURCE_PATH = "src/openpi/models_pytorch/draft.py"
SOURCE_SHA256 = "186deb7f6cf89c3476b1b5f6d37350dc7a6073c55367d4edb99c89dea8c3793d"

BASE_MODEL_FAMILY = "pi0"
PUBLISHED_DRAFT_ARCH = "vlm1_query_decoder"
PINNED_TRAIN_SCRIPT_ARCH = "vlm_block"
PUBLISHED_SAMPLE_SEMANTICS = "sliding_chunk_shift_v2"
SOURCE_DIVERGENCES = (
    "published checkpoint meta draft_arch='vlm1_query_decoder', while "
    "spec_draft_train.py at SOURCE_REVISION emits draft_arch='vlm_block'",
    "the compatibility path supplies head_dim/rope settings explicitly to "
    "pre-config-API GemmaRotaryEmbedding releases; the rotary math is unchanged",
)

_EXPECTED_META: Mapping[str, object] = MappingProxyType(
    {
        "chunk_m": 50,
        "out_dim": 7,
        "img_dim": 2048,
        "draft_arch": PUBLISHED_DRAFT_ARCH,
        "draft_input_mode": "prefix_embs",
        "draft_hidden_size": 2048,
        "draft_num_heads": 8,
        "draft_num_kv_heads": 1,
        "draft_head_dim": 256,
        "use_last_actions": False,
        "use_seed_actions": False,
        "sample_semantics": PUBLISHED_SAMPLE_SEMANTICS,
    }
)

_EXPECTED_STATE_SHAPES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "_state_token.weight": (2048, 32),
        "_state_token.bias": (2048,),
        "_action_queries.weight": (50, 2048),
        "_gemma_block.self_attn.q_proj.weight": (2048, 2048),
        "_gemma_block.self_attn.k_proj.weight": (256, 2048),
        "_gemma_block.self_attn.v_proj.weight": (256, 2048),
        "_gemma_block.self_attn.o_proj.weight": (2048, 2048),
        "_gemma_block.mlp.gate_proj.weight": (16384, 2048),
        "_gemma_block.mlp.up_proj.weight": (16384, 2048),
        "_gemma_block.mlp.down_proj.weight": (2048, 16384),
        "_gemma_block.input_layernorm.weight": (2048,),
        "_gemma_block.post_attention_layernorm.weight": (2048,),
        "_action_head.weight": (7, 2048),
        "_action_head.bias": (7,),
    }
)
_EXPECTED_STATE_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {
        name: (
            torch.bfloat16
            if name
            in {
                "_gemma_block.input_layernorm.weight",
                "_gemma_block.post_attention_layernorm.weight",
            }
            else torch.float32
        )
        for name in _EXPECTED_STATE_SHAPES
    }
)
EXPECTED_PARAMETER_COUNT = 110_288_903


class DraftChunkHead(nn.Module):
    """One-layer Gemma query decoder over pi0 prefix embeddings.

    The computation and state-dict names intentionally match the pinned
    Dexmal implementation.  A newly constructed module is initialization-only
    until :func:`load_draft_chunk_head` binds a verified production resolution.
    """

    def __init__(
        self,
        *,
        img_dim: int,
        chunk_m: int,
        hidden_dim: int = 256,
        out_dim: int = 7,
        num_heads: int | None = None,
        num_kv_heads: int = 1,
        head_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        self.chunk_m = int(chunk_m)
        self.out_dim = int(out_dim)
        self.pose_rot_dim = int(min(6, self.out_dim))
        self.hidden_size = int(img_dim)
        self.num_heads = int(num_heads or self._resolve_num_heads(self.hidden_size))
        self.num_kv_heads = int(max(1, int(num_kv_heads)))
        self.head_dim = int(head_dim or max(1, self.hidden_size // self.num_heads))
        self.attn_implementation = str(attn_implementation)

        gemma_config = CONFIG_MAPPING["gemma"](
            head_dim=int(self.head_dim),
            hidden_size=int(self.hidden_size),
            intermediate_size=int(hidden_dim),
            num_attention_heads=int(self.num_heads),
            num_hidden_layers=1,
            num_key_value_heads=int(self.num_kv_heads),
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype=str(dtype).replace("torch.", ""),
        )
        gemma_config._attn_implementation = self.attn_implementation  # noqa: SLF001

        self._state_token = nn.Linear(32, int(self.hidden_size))
        self._action_queries = nn.Embedding(int(self.chunk_m), int(self.hidden_size))
        self._gemma_block = GemmaDecoderLayer(gemma_config, layer_idx=0)
        try:
            self._rotary_emb = GemmaRotaryEmbedding(gemma_config)
        except TypeError:
            # Transformers 4.47 uses the older ``(dim, max_position, base)``
            # constructor.  These are the same values read from GemmaConfig by
            # newer releases, so checkpoint tensor geometry and rotary math do
            # not change.
            self._rotary_emb = GemmaRotaryEmbedding(
                int(self.head_dim),
                max_position_embeddings=int(gemma_config.max_position_embeddings),
                base=int(gemma_config.rope_theta),
            )
        self._action_head = nn.Linear(int(self.hidden_size), int(self.out_dim))

        self._initialization_only = True
        self._production_compatible = False
        self._checkpoint_metadata: Mapping[str, Any] = MappingProxyType({})
        self._checkpoint_path: Path | None = None

    @staticmethod
    def _resolve_num_heads(dim: int) -> int:
        for heads in (8, 4, 2, 1):
            if dim % heads == 0:
                return heads
        return 1

    @staticmethod
    def _make_att_2d_masks(
        pad_masks: torch.Tensor, att_masks: torch.Tensor
    ) -> torch.Tensor:
        cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def _build_attention_mask(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prefix_pad_masks.ndim != 2:
            raise ValueError(
                "expected prefix_pad_masks to be (B,S), "
                f"got shape={tuple(prefix_pad_masks.shape)}"
            )
        batch, sequence = (
            int(prefix_pad_masks.shape[0]),
            int(prefix_pad_masks.shape[1]),
        )
        device = prefix_pad_masks.device
        if prefix_att_masks is None:
            prefix_att_masks = torch.zeros(
                (batch, sequence), device=device, dtype=torch.bool
            )
        if prefix_att_masks.ndim != 2 or tuple(prefix_att_masks.shape) != tuple(
            prefix_pad_masks.shape
        ):
            raise ValueError(
                "expected prefix_att_masks to match prefix_pad_masks "
                f"shape={tuple(prefix_pad_masks.shape)}, "
                f"got {tuple(prefix_att_masks.shape)}"
            )

        state_pad = torch.ones((batch, 1), device=device, dtype=torch.bool)
        state_att = torch.zeros((batch, 1), device=device, dtype=torch.bool)
        prefix_plus_state_pad = torch.cat(
            [prefix_pad_masks.to(dtype=torch.bool), state_pad], dim=1
        )
        prefix_plus_state_att = torch.cat(
            [prefix_att_masks.to(dtype=torch.bool), state_att], dim=1
        )
        prefix_mask = self._make_att_2d_masks(
            prefix_plus_state_pad, prefix_plus_state_att
        )

        query_count = int(self.chunk_m)
        total = int(prefix_mask.shape[1] + query_count)
        mask = torch.zeros((batch, total, total), device=device, dtype=torch.bool)
        prefix_len = int(prefix_mask.shape[1])
        mask[:, :prefix_len, :prefix_len] = prefix_mask
        mask[:, prefix_len:, :prefix_len] = prefix_plus_state_pad[:, None, :]
        mask[:, prefix_len:, prefix_len:] = True
        return mask

    def _build_position_ids(self, *, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        batch = int(prefix_pad_masks.shape[0])
        state_pad = torch.ones(
            (batch, 1), device=prefix_pad_masks.device, dtype=torch.bool
        )
        query_pad = torch.ones(
            (batch, int(self.chunk_m)),
            device=prefix_pad_masks.device,
            dtype=torch.bool,
        )
        pad_mask = torch.cat(
            [prefix_pad_masks.to(dtype=torch.bool), state_pad, query_pad], dim=1
        )
        return (torch.cumsum(pad_mask.to(dtype=torch.int64), dim=1) - 1).clamp_min(0)

    @property
    def initialization_only(self) -> bool:
        """Whether the module is prohibited from direct production inference."""

        return self._initialization_only

    @property
    def production_compatible(self) -> bool:
        """Whether the verified Draft resolution authorized production use."""

        return self._production_compatible

    @property
    def base_model_family(self) -> str:
        """Return the immutable source policy family (pi0, never pi0.5)."""

        return BASE_MODEL_FAMILY

    @property
    def pi05_compatible(self) -> bool:
        """Return false: published pi0 Draft weights are not pi0.5 weights."""

        return False

    @property
    def checkpoint_metadata(self) -> Mapping[str, Any]:
        """Return the read-only checkpoint metadata recorded at load time."""

        return self._checkpoint_metadata

    @property
    def checkpoint_path(self) -> Path | None:
        """Return the verified source checkpoint path, if loaded."""

        return self._checkpoint_path

    def init_from_vlm_layer(self, layer: nn.Module) -> None:
        """Warm-start the Gemma block from a shape-compatible VLM layer."""

        self._gemma_block.load_state_dict(layer.state_dict(), strict=True)

    def production_predict_adapter(self) -> "DraftProductionPredictAdapter":
        """Build a guarded production adapter or fail closed."""

        if self.initialization_only or not self.production_compatible:
            raise ConfigurationError(
                "initialization-only Draft head cannot create a production "
                "predict adapter"
            )
        return DraftProductionPredictAdapter(self)

    def _bind_resolution(
        self,
        *,
        production_compatible: bool,
        metadata: Mapping[str, Any],
        checkpoint_path: Path,
    ) -> None:
        self._production_compatible = bool(production_compatible)
        self._initialization_only = not self._production_compatible
        self._checkpoint_metadata = MappingProxyType(dict(metadata))
        self._checkpoint_path = checkpoint_path

    def forward(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        robot_state: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one 50-step pi0 action chunk from cached VLM embeddings."""

        del last_actions
        if prefix_embs.ndim != 3:
            raise ValueError(
                "expected prefix_embs to be (B,S,H), "
                f"got shape={tuple(prefix_embs.shape)}"
            )
        if prefix_pad_masks.ndim != 2:
            raise ValueError(
                "expected prefix_pad_masks to be (B,S), "
                f"got shape={tuple(prefix_pad_masks.shape)}"
            )
        if prefix_att_masks.ndim != 2:
            raise ValueError(
                "expected prefix_att_masks to be (B,S), "
                f"got shape={tuple(prefix_att_masks.shape)}"
            )
        if robot_state.ndim != 2 or int(robot_state.shape[1]) != 32:
            raise ValueError(
                "expected robot_state to be (B,32), "
                f"got shape={tuple(robot_state.shape)}"
            )
        if int(prefix_embs.shape[0]) != int(robot_state.shape[0]):
            raise ValueError(
                "prefix_embs and robot_state must have matching batch dimensions"
            )
        if int(prefix_embs.shape[1]) != int(prefix_pad_masks.shape[1]) or int(
            prefix_embs.shape[1]
        ) != int(prefix_att_masks.shape[1]):
            raise ValueError(
                "prefix_embs, prefix_pad_masks, and prefix_att_masks must have "
                "matching sequence lengths"
            )
        if int(prefix_embs.shape[2]) != int(self.hidden_size):
            raise ValueError(
                f"expected prefix_embs hidden size={self.hidden_size}, "
                f"got {int(prefix_embs.shape[2])}"
            )

        batch = int(robot_state.shape[0])
        block_dtype = self._gemma_block.self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=block_dtype)
        state_token = self._state_token(
            robot_state.to(dtype=self._state_token.weight.dtype)
        )[:, None, :].to(dtype=block_dtype)
        query_ids = torch.arange(
            int(self.chunk_m), device=prefix_embs.device, dtype=torch.long
        )[None, :].expand(batch, -1)
        query_tokens = self._action_queries(query_ids).to(dtype=block_dtype)
        hidden_states = torch.cat([prefix_embs, state_token, query_tokens], dim=1)

        mask_2d = self._build_attention_mask(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
        )
        attention_mask = torch.where(
            mask_2d[:, None, :, :],
            torch.zeros((), device=hidden_states.device, dtype=block_dtype),
            torch.full(
                (),
                torch.finfo(block_dtype).min,
                device=hidden_states.device,
                dtype=block_dtype,
            ),
        )
        position_ids = self._build_position_ids(prefix_pad_masks=prefix_pad_masks)
        position_embeddings = self._rotary_emb(hidden_states, position_ids)
        hidden_states = self._gemma_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
            adarms_cond=None,
        )[0]

        query_hidden = hidden_states[:, -int(self.chunk_m) :, :].to(
            dtype=self._action_head.weight.dtype
        )
        return self._action_head(query_hidden).to(dtype=torch.float32)


class DraftProductionPredictAdapter:
    """Guarded tensor-level production callable for a verified Draft head."""

    def __init__(self, head: DraftChunkHead) -> None:
        if head.initialization_only or not head.production_compatible:
            raise ConfigurationError(
                "initialization-only Draft head cannot enter production inference"
            )
        head.eval()
        head.requires_grad_(False)
        self._head = head

    @property
    def head(self) -> DraftChunkHead:
        """Return the source-locked Draft module."""

        return self._head

    def predict(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Run the Draft head after rechecking its production authorization."""

        if self._head.initialization_only or not self._head.production_compatible:
            raise ConfigurationError("Draft production authorization was revoked")
        if self._head.training or any(
            parameter.requires_grad for parameter in self._head.parameters()
        ):
            raise ConfigurationError(
                "production Draft head must remain in eval mode with frozen parameters"
            )
        with torch.inference_mode():
            return self._head(**inputs)

    __call__: Callable[..., torch.Tensor] = predict


def _require_meta(meta: Any) -> Mapping[str, Any]:
    if not isinstance(meta, Mapping):
        raise ConfigurationError("Draft checkpoint meta must be an object")
    for key, expected in _EXPECTED_META.items():
        if key not in meta:
            raise ConfigurationError(f"Draft checkpoint meta is missing {key!r}")
        observed = meta[key]
        if isinstance(expected, bool):
            type_matches = isinstance(observed, bool)
        elif isinstance(expected, int):
            type_matches = isinstance(observed, int) and not isinstance(observed, bool)
        else:
            type_matches = isinstance(observed, type(expected))
        if not type_matches or observed != expected:
            raise ConfigurationError(
                f"Draft checkpoint meta {key!r} mismatch: "
                f"expected {expected!r}, got {observed!r}"
            )
    for family_key in ("model_family", "base_model_family"):
        if family_key in meta and meta[family_key] != BASE_MODEL_FAMILY:
            raise ConfigurationError(
                f"published Draft checkpoint must remain {BASE_MODEL_FAMILY}; "
                f"meta {family_key!r} cannot be {meta[family_key]!r}"
            )
    return meta


def _require_state_dict(state_dict: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping):
        raise ConfigurationError("Draft checkpoint draft_head must be a state_dict")
    observed_keys = set(state_dict)
    expected_keys = set(_EXPECTED_STATE_SHAPES)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ConfigurationError(
            "Draft checkpoint state_dict schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    parameter_count = 0
    for name, expected_shape in _EXPECTED_STATE_SHAPES.items():
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ConfigurationError(f"Draft checkpoint {name!r} is not a tensor")
        observed_shape = tuple(int(value) for value in tensor.shape)
        if observed_shape != expected_shape:
            raise ConfigurationError(
                f"Draft checkpoint {name!r} shape mismatch: "
                f"expected {expected_shape}, got {observed_shape}"
            )
        expected_dtype = _EXPECTED_STATE_DTYPES[name]
        if tensor.dtype != expected_dtype:
            raise ConfigurationError(
                f"Draft checkpoint {name!r} dtype mismatch: "
                f"expected {expected_dtype}, got {tensor.dtype}"
            )
        parameter_count += tensor.numel()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ConfigurationError(
            "Draft checkpoint parameter count mismatch: "
            f"expected {EXPECTED_PARAMETER_COUNT}, got {parameter_count}"
        )
    return state_dict


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        # Old Torch cannot use the restricted unpickler.  Falling back is
        # permitted only because DraftResolution bytes were SHA-256 verified
        # immediately before this call.
        warnings.warn(
            "installed Torch lacks weights_only loading; falling back only for "
            "the SHA-256-verified published Draft checkpoint",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu")


def _require_resolution(
    resolution: DraftResolution, purpose: Literal["production", "warm_start"]
) -> bool:
    if not isinstance(resolution, DraftResolution):
        raise TypeError("resolution must be a draft_assets.DraftResolution")
    compatibility = resolution.compatibility
    if purpose == "production":
        if (
            not compatibility.production_compatible
            or compatibility.initialization_allowed
            or compatibility.purpose != "production"
        ):
            raise ConfigurationError(
                "DraftResolution does not authorize production inference"
            )
        return True
    if purpose == "warm_start":
        if (
            compatibility.production_compatible
            or not compatibility.initialization_allowed
            or compatibility.purpose != "warm_start_initialization_only"
        ):
            raise ConfigurationError(
                "DraftResolution does not authorize initialization-only warm start"
            )
        return False
    raise ConfigurationError("purpose must be 'production' or 'warm_start'")


def load_draft_chunk_head(
    resolution: DraftResolution,
    *,
    purpose: Literal["production", "warm_start"],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DraftChunkHead:
    """Load one verified published Draft head under an explicit use policy.

    Args:
        resolution: Content-verified asset and semantic compatibility decision.
        purpose: ``production`` for an exact pi0 LIBERO target, or
            ``warm_start`` for retraining initialization only.
        device: Destination Torch device.
        dtype: Runtime module dtype.  Checkpoint tensor dtypes are validated
            before this conversion.

    Returns:
        A source-equivalent :class:`DraftChunkHead` carrying immutable-facing
        compatibility properties.
    """

    production_compatible = _require_resolution(resolution, purpose)
    # Re-verify immediately before deserialization; a stale VerifiedDraftAsset
    # object is not treated as an eternal proof about mutable local bytes.
    verified = resolution.verified_asset.asset.verify(resolution.path.parent)
    if verified.path != resolution.path.resolve():
        raise ConfigurationError("Draft resolution path changed after verification")

    checkpoint = _safe_torch_load(verified.path)
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "draft_head",
        "meta",
    }:
        raise ConfigurationError(
            "Draft checkpoint must contain exactly 'draft_head' and 'meta'"
        )
    meta = _require_meta(checkpoint["meta"])
    state_dict = _require_state_dict(checkpoint["draft_head"])

    target_device = torch.device(device)
    with torch.device(target_device):
        head = DraftChunkHead(
            img_dim=2048,
            chunk_m=50,
            hidden_dim=16384,
            out_dim=7,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            dtype=dtype,
            attn_implementation="sdpa",
        )
    head = head.to(device=target_device, dtype=dtype)
    incompatible = head.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ConfigurationError(
            "strict Draft state_dict load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if sum(parameter.numel() for parameter in head.parameters()) != (
        EXPECTED_PARAMETER_COUNT
    ):
        raise ConfigurationError("constructed Draft head parameter count changed")

    head._bind_resolution(  # noqa: SLF001
        production_compatible=production_compatible,
        metadata=meta,
        checkpoint_path=verified.path,
    )
    if production_compatible:
        head.eval()
        head.requires_grad_(False)
    return head


__all__ = [
    "BASE_MODEL_FAMILY",
    "EXPECTED_PARAMETER_COUNT",
    "PINNED_TRAIN_SCRIPT_ARCH",
    "PUBLISHED_DRAFT_ARCH",
    "PUBLISHED_SAMPLE_SEMANTICS",
    "SOURCE_DIVERGENCES",
    "SOURCE_PATH",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_SHA256",
    "DraftChunkHead",
    "DraftProductionPredictAdapter",
    "load_draft_chunk_head",
]
