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

"""Checkpoint-bound OpenPI prefix preparation and parallel verification.

This module owns the boundary between an already loaded OpenPI target and a
continuous-action draft.  All tensors handled here are in normalized OpenPI
model space.  In particular, no environment-space action is silently padded
to the model width, and no PI0 draft is treated as compatible with PI0.5 just
because its tensor shape happens to match.

The implementation follows the audited SpecPI0 flow convention:

``x_t = t * noise + (1 - t) * x0_draft``
``x0_hat = x_t - t * v_theta(x_t, t)``

The encoder and VLM prefill run once in :meth:`prepare_prefix`.  Verification
then expands the normalized state, masks, and KV cache from ``B`` to ``B*K``
and invokes exactly one target velocity/denoiser call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from ..errors import ConfigurationError, ShapeMismatchError

_VERIFIED_OPENPI_TARGET_PROOF = object()
_VERIFIED_DRAFT_PREDICTION_PROOF = object()

PolicyFamily = Literal["pi0", "pi0.5"]
VelocityTarget = Callable[
    [torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor], torch.Tensor
]


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _ensure_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _ensure_floating_finite(tensor: Any, name: str, *, ndim: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim:
        raise ShapeMismatchError(f"{name} must be a rank-{ndim} Torch tensor")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN or infinity")
    return tensor


def _ensure_mask(tensor: Any, name: str, *, shape: tuple[int, int]) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise ShapeMismatchError(f"{name} must have shape {shape}")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if tensor.dtype != torch.bool and tensor.dtype not in integer_dtypes:
        raise TypeError(f"{name} must use a boolean or integer dtype")
    if tensor.dtype != torch.bool and bool(
        ((tensor != 0) & (tensor != 1)).any().item()
    ):
        raise ValueError(f"{name} must contain only boolean 0/1 mask values")
    return tensor


@dataclass(frozen=True)
class OpenPITargetContract:
    """Exact model-space lineage a draft was trained to target.

    ``checkpoint_id`` should be a revision- or manifest-bound identity rather
    than a mutable directory name.  ``norm_stats_id`` and ``transform_id`` are
    separate on purpose: equal checkpoint shapes do not make two robot action
    semantics interchangeable.  ``env_action_indices`` is the only supported
    model-to-environment dimension selection; an implicit leading slice is
    deliberately forbidden.
    """

    policy_family: PolicyFamily
    checkpoint_id: str
    config_name: str
    norm_stats_id: str
    transform_id: str
    action_horizon: int
    model_action_dim: int
    env_action_indices: tuple[int, ...]
    state_dim: int = 32
    camera_keys: tuple[str, ...] = ()
    model_camera_keys: tuple[str, ...] = ()
    environment_action_semantics: tuple[str, ...] = ()
    delta_action_mask: tuple[bool, ...] = ()
    max_token_length: int = 0
    normalization: str = ""
    asset_id: str = ""

    def __post_init__(self) -> None:
        if self.policy_family not in ("pi0", "pi0.5"):
            raise ValueError("policy_family must be 'pi0' or 'pi0.5'")
        for name in (
            "checkpoint_id",
            "config_name",
            "norm_stats_id",
            "transform_id",
        ):
            object.__setattr__(self, name, _ensure_nonempty(getattr(self, name), name))
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.model_action_dim <= 0:
            raise ValueError("model_action_dim must be positive")
        if self.state_dim != 32:
            raise ValueError(
                "OpenPI speculative state_dim must be the padded 32D model state"
            )
        if not self.env_action_indices:
            raise ValueError(
                "env_action_indices must explicitly select environment actions"
            )
        if len(set(self.env_action_indices)) != len(self.env_action_indices):
            raise ValueError("env_action_indices must be unique")
        if any(
            index < 0 or index >= self.model_action_dim
            for index in self.env_action_indices
        ):
            raise ValueError("env_action_indices exceed the OpenPI model action width")
        if any(not isinstance(key, str) or not key.strip() for key in self.camera_keys):
            raise ValueError("camera_keys must contain non-empty strings")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be unique")
        if self.model_camera_keys and len(self.model_camera_keys) != len(
            self.camera_keys
        ):
            raise ValueError("model_camera_keys must align with camera_keys")
        if len(set(self.model_camera_keys)) != len(self.model_camera_keys):
            raise ValueError("model_camera_keys must be unique")
        environment_dim = len(self.env_action_indices)
        if (
            self.environment_action_semantics
            and len(self.environment_action_semantics) != environment_dim
        ):
            raise ValueError(
                "environment_action_semantics must align with env_action_indices"
            )
        if self.delta_action_mask and len(self.delta_action_mask) != environment_dim:
            raise ValueError("delta_action_mask must align with env_action_indices")
        if self.max_token_length < 0:
            raise ValueError("max_token_length must be non-negative")

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        """Return fields that define the exact draft/target semantic space."""

        return (
            self.policy_family,
            self.checkpoint_id,
            self.config_name,
            self.norm_stats_id,
            self.transform_id,
            self.action_horizon,
            self.model_action_dim,
            self.env_action_indices,
            self.state_dim,
            self.camera_keys,
            self.model_camera_keys,
            self.environment_action_semantics,
            self.delta_action_mask,
            self.max_token_length,
            self.normalization,
            self.asset_id,
        )

    def require_compatible_draft(self, draft_target: "OpenPITargetContract") -> None:
        """Reject a draft that was not trained for this exact target space."""

        if not isinstance(draft_target, OpenPITargetContract):
            raise TypeError("draft_target_contract must be an OpenPITargetContract")
        if draft_target.compatibility_key != self.compatibility_key:
            mismatches = tuple(
                name
                for name in (
                    "policy_family",
                    "checkpoint_id",
                    "config_name",
                    "norm_stats_id",
                    "transform_id",
                    "action_horizon",
                    "model_action_dim",
                    "env_action_indices",
                    "state_dim",
                    "camera_keys",
                    "model_camera_keys",
                    "environment_action_semantics",
                    "delta_action_mask",
                    "max_token_length",
                    "normalization",
                    "asset_id",
                )
                if getattr(draft_target, name) != getattr(self, name)
            )
            raise ConfigurationError(
                "draft target is incompatible with the loaded OpenPI target: "
                + ", ".join(mismatches)
            )


def _sha256_identity(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _contract_from_verified_assets(verified_assets: Any) -> OpenPITargetContract:
    from ..assets import require_verified_checkpoint_assets

    assets = require_verified_checkpoint_assets(verified_assets)
    manifest = assets.manifest
    runtime = manifest.runtime
    norm_file = next(
        (file for file in manifest.files if file.role == "selected_norm_stats"),
        None,
    )
    if norm_file is None or norm_file.path != runtime.norm_stats_path:
        raise ConfigurationError(
            "verified checkpoint lacks its selected norm-stats manifest entry"
        )
    weight_files = tuple(
        file for file in manifest.files if file.role == "weights_shard"
    )
    if not weight_files:
        raise ConfigurationError("verified checkpoint has no weight shards")

    weight_identity = _sha256_identity(
        {
            "files": [
                {"path": file.path.as_posix(), "sha256": file.sha256}
                for file in weight_files
            ]
        }
    )
    transform_identity = _sha256_identity(
        {
            "model_family": runtime.model_family,
            "config_name": runtime.config_name,
            "asset_id": runtime.asset_id,
            "action_mode": runtime.action_mode,
            "policy_frame": runtime.policy_frame,
            "raw_state_dim": runtime.state_dim,
            "model_state_dim": runtime.model_state_dim,
            "state_semantics": runtime.state_semantics,
            "environment_action_semantics": runtime.environment_action_semantics,
            "delta_action_mask": runtime.delta_action_mask,
            "extra_delta_transform": runtime.extra_delta_transform,
            "adapt_to_pi": runtime.adapt_to_pi,
            "camera_keys": runtime.cameras.observation_keys,
            "model_camera_keys": runtime.cameras.model_image_keys,
            "sources": [
                {
                    "path": source.path.as_posix(),
                    "sha256_lf": source.sha256_lf,
                    "repository": source.repository,
                    "revision": source.revision,
                }
                for source in runtime.source_fingerprints
            ],
        }
    )
    family: PolicyFamily
    normalized_family = runtime.model_family.lower().replace("_", "")
    if normalized_family in {"pi05", "pi0.5"}:
        family = "pi0.5"
    elif normalized_family == "pi0":
        family = "pi0"
    else:
        raise ConfigurationError(
            f"unsupported verified OpenPI model family {runtime.model_family!r}"
        )
    return OpenPITargetContract(
        policy_family=family,
        checkpoint_id=f"{assets.checkpoint_id}#weights-sha256:{weight_identity}",
        config_name=runtime.config_name,
        norm_stats_id=(
            f"{runtime.asset_id}/{runtime.norm_stats_path.as_posix()}"
            f"@sha256:{norm_file.sha256}"
        ),
        transform_id=f"sha256:{transform_identity}",
        action_horizon=runtime.action_horizon,
        model_action_dim=runtime.model_action_dim,
        env_action_indices=tuple(range(runtime.environment_action_dim)),
        state_dim=runtime.model_state_dim,
        camera_keys=runtime.cameras.observation_keys,
        model_camera_keys=runtime.cameras.model_image_keys,
        environment_action_semantics=runtime.environment_action_semantics,
        delta_action_mask=runtime.delta_action_mask,
        max_token_length=runtime.max_token_length,
        normalization=runtime.normalization,
        asset_id=runtime.asset_id,
    )


def _require_exact_model_load(model: Any, verified_assets: Any) -> None:
    report = getattr(model, "_rlinf_checkpoint_load_report", None)
    if not isinstance(report, Mapping):
        raise ConfigurationError(
            "loaded OpenPI target lacks the RLinf checkpoint load report"
        )
    if report.get("source_kind") != "safetensors_shards":
        raise ConfigurationError(
            "loaded OpenPI target was not selected from verified safetensors shards"
        )
    from ..assets import checkpoint_load_key_classes

    try:
        missing, unexpected, ignored_auxiliary, unresolved = (
            checkpoint_load_key_classes(dict(report))
        )
    except ValueError as error:
        raise ConfigurationError(
            f"loaded OpenPI target has an invalid checkpoint report: {error}"
        ) from error
    if missing or unresolved:
        raise ConfigurationError(
            "loaded OpenPI target main state_dict is not exact-key compatible: "
            f"missing={missing}, raw_unexpected={unexpected}, "
            f"ignored_auxiliary={ignored_auxiliary}, unresolved={unresolved}"
        )
    selected = {
        Path(path).expanduser().resolve() for path in report.get("selected_paths", ())
    }
    expected = {path.resolve() for path in verified_assets.weight_paths}
    if selected != expected:
        raise ConfigurationError(
            "loaded OpenPI target paths differ from the verified weight shards"
        )


@dataclass(frozen=True)
class VerifiedOpenPITarget:
    """Loaded model bound to verified bytes and an immutable semantic contract."""

    model: Any
    assets: Any
    contract: OpenPITargetContract
    _target_token: object = field(repr=False, compare=False)
    _verification_proof: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._verification_proof is not _VERIFIED_OPENPI_TARGET_PROOF:
            raise ConfigurationError(
                "VerifiedOpenPITarget must be issued from verified checkpoint assets"
            )

    @classmethod
    def from_verified_assets(
        cls,
        model: Any,
        verified_assets: Any,
    ) -> "VerifiedOpenPITarget":
        """Bind one exact RLinf load to its verified manifest and runtime geometry."""

        from ..assets import require_verified_checkpoint_assets

        assets = require_verified_checkpoint_assets(verified_assets)
        _require_exact_model_load(model, assets)
        contract = _contract_from_verified_assets(assets)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        bound = cls(
            model=model,
            assets=assets,
            contract=contract,
            _target_token=object(),
            _verification_proof=_VERIFIED_OPENPI_TARGET_PROOF,
        )
        bound.validate_runtime_model()
        return bound

    def validate_runtime_model(self) -> None:
        """Recheck model geometry and inference-only state before every binding."""

        if self._verification_proof is not _VERIFIED_OPENPI_TARGET_PROOF:
            raise ConfigurationError("OpenPI target verification proof is invalid")
        config = getattr(self.model, "config", None)
        if config is None:
            raise ConfigurationError("OpenPI target must expose model.config")
        expected = self.contract
        exact_fields = {
            "config_name": expected.config_name,
            "action_horizon": expected.action_horizon,
            "action_chunk": expected.action_horizon,
            "action_dim": expected.model_action_dim,
            "action_env_dim": len(expected.env_action_indices),
            "max_token_len": expected.max_token_length,
            "num_images_in_input": len(expected.camera_keys),
            "pi05": expected.policy_family == "pi0.5",
        }
        mismatches = []
        for name, expected_value in exact_fields.items():
            observed = getattr(config, name, None)
            if observed != expected_value:
                mismatches.append(
                    f"{name}: loaded={observed!r}, verified={expected_value!r}"
                )
        if mismatches:
            raise ConfigurationError(
                "loaded OpenPI target conflicts with verified assets: "
                + "; ".join(mismatches)
            )
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise ConfigurationError(
                "verified OpenPI inference target must remain eval-mode and frozen"
            )


@dataclass(frozen=True)
class VerifiedDraftPrediction:
    """Model-space Draft output issued by a verified production Draft loader."""

    actions_model: torch.Tensor
    target_contract: OpenPITargetContract
    draft_checkpoint_id: str
    draft_checkpoint_sha256: str
    _target_token: object = field(repr=False, compare=False)
    _verification_proof: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._verification_proof is not _VERIFIED_DRAFT_PREDICTION_PROOF:
            raise ConfigurationError(
                "VerifiedDraftPrediction must be issued by a verified Draft loader"
            )
        _ensure_nonempty(self.draft_checkpoint_id, "draft_checkpoint_id")
        digest = _ensure_nonempty(
            self.draft_checkpoint_sha256, "draft_checkpoint_sha256"
        ).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ConfigurationError("draft_checkpoint_sha256 must be a SHA-256 hex")


def _issue_verified_draft_prediction(
    *,
    actions_model: torch.Tensor,
    target: VerifiedOpenPITarget,
    target_contract: OpenPITargetContract,
    draft_checkpoint_id: str,
    draft_checkpoint_sha256: str,
) -> VerifiedDraftPrediction:
    """Internal issuance hook used only after a Draft loader verifies its manifest."""

    if not isinstance(target, VerifiedOpenPITarget):
        raise TypeError("target must be a VerifiedOpenPITarget")
    target.validate_runtime_model()
    target.contract.require_compatible_draft(target_contract)
    return VerifiedDraftPrediction(
        actions_model=actions_model,
        target_contract=target_contract,
        draft_checkpoint_id=draft_checkpoint_id,
        draft_checkpoint_sha256=draft_checkpoint_sha256,
        _target_token=target._target_token,
        _verification_proof=_VERIFIED_DRAFT_PREDICTION_PROOF,
    )


@dataclass(frozen=True)
class PrefixBundle:
    """One contextualized OpenPI prefix owned by one verifier episode."""

    prefix_embs: torch.Tensor
    prefix_pad_masks: torch.Tensor
    prefix_att_masks: torch.Tensor
    normalized_state: torch.Tensor
    contextual_prefix_output: torch.Tensor
    past_key_values: Any
    episode_id: str
    target_contract: OpenPITargetContract
    _owner_token: object = field(repr=False, compare=False)
    _generation: int = field(repr=False, compare=False)

    @property
    def batch_size(self) -> int:
        """Return the observation batch size."""

        return int(self.normalized_state.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device shared by all model-space prefix tensors."""

        return self.normalized_state.device

    @property
    def kv_cache_type(self) -> str:
        """Return the exact prefill cache type for audit diagnostics."""

        return _qualified_type(self.past_key_values)


@dataclass(frozen=True)
class OpenPIParallelVerification:
    """Target velocity and clean-action reconstruction from one ``B*K`` call."""

    draft_model: torch.Tensor
    shared_noise: torch.Tensor
    verification_times: torch.Tensor
    interpolation_model: torch.Tensor
    predicted_velocity_model: torch.Tensor
    x0_hat_model: torch.Tensor
    draft_env: torch.Tensor
    shared_noise_env: torch.Tensor
    predicted_velocity_env: torch.Tensor
    x0_hat_env: torch.Tensor
    env_action_indices: tuple[int, ...]
    velocity_path: str
    kv_cache_type: str
    expanded_kv_cache_type: str
    episode_id: str
    target_contract: OpenPITargetContract
    draft_checkpoint_id: str
    draft_checkpoint_sha256: str

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return stable evidence about the executed verification path."""

        batch_size, verification_count, horizon, model_dim = self.x0_hat_model.shape
        return {
            "batch_size": int(batch_size),
            "verification_count": int(verification_count),
            "action_horizon": int(horizon),
            "model_action_dim": int(model_dim),
            "env_action_indices": self.env_action_indices,
            "velocity_path": self.velocity_path,
            "velocity_calls": 1,
            "encoder_prefill_reused": True,
            "shared_noise_across_k": True,
            "normalized_model_space": True,
            "kv_cache_type": self.kv_cache_type,
            "expanded_kv_cache_type": self.expanded_kv_cache_type,
            "episode_id": self.episode_id,
            "policy_family": self.target_contract.policy_family,
            "checkpoint_id": self.target_contract.checkpoint_id,
            "draft_checkpoint_id": self.draft_checkpoint_id,
            "draft_checkpoint_sha256": self.draft_checkpoint_sha256,
            "draft_prediction_asset_verified": True,
        }


class _LegacyCacheFactory(Protocol):
    @classmethod
    def from_legacy_cache(cls, cache: Any) -> Any:
        """Rebuild one cache while preserving its concrete type."""


def _cache_as_legacy(cache: Any) -> tuple[Sequence[Any], type[Any] | None]:
    if isinstance(cache, (tuple, list)):
        return cache, None
    to_legacy = getattr(cache, "to_legacy_cache", None)
    factory = getattr(type(cache), "from_legacy_cache", None)
    if not callable(to_legacy) or not callable(factory):
        raise TypeError(
            "past_key_values must be a legacy list/tuple or expose "
            "to_legacy_cache/from_legacy_cache"
        )
    legacy = to_legacy()
    if not isinstance(legacy, (tuple, list)):
        raise TypeError("KV cache to_legacy_cache() must return a list or tuple")
    return legacy, type(cache)


def _validate_legacy_cache(
    legacy: Sequence[Any], *, batch_size: int, device: torch.device
) -> None:
    if not legacy:
        raise ShapeMismatchError("past_key_values must contain at least one layer")
    for layer_index, layer in enumerate(legacy):
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise TypeError(f"KV layer {layer_index} must be a key/value pair")
        key, value = layer
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ShapeMismatchError(
                f"KV layer {layer_index} key/value must be Torch tensors"
            )
        if key.shape != value.shape or key.dtype != value.dtype:
            raise ShapeMismatchError(
                f"KV layer {layer_index} key/value shape or dtype differs"
            )
        for value_name, value in zip(("key", "value"), layer, strict=True):
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise ShapeMismatchError(
                    f"KV layer {layer_index} {value_name} must be a Torch tensor"
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"KV layer {layer_index} {value_name} must use a floating dtype"
                )
            if int(value.shape[0]) != batch_size:
                raise ShapeMismatchError(
                    f"KV layer {layer_index} {value_name} batch does not match prefix"
                )
            if value.device != device:
                raise ValueError(
                    f"KV layer {layer_index} {value_name} is on a different device"
                )
            if torch.is_floating_point(value) and not bool(
                torch.isfinite(value).all().item()
            ):
                raise ValueError(
                    f"KV layer {layer_index} {value_name} contains NaN or infinity"
                )


def _validate_cache(cache: Any, *, batch_size: int, device: torch.device) -> None:
    legacy, _ = _cache_as_legacy(cache)
    _validate_legacy_cache(legacy, batch_size=batch_size, device=device)


def _expand_batch(tensor: torch.Tensor, count: int) -> torch.Tensor:
    if count == 1:
        return tensor
    batch_size = int(tensor.shape[0])
    return (
        tensor.unsqueeze(1)
        .expand(batch_size, count, *tensor.shape[1:])
        .reshape(batch_size * count, *tensor.shape[1:])
    )


def _expand_kv_cache(cache: Any, count: int, *, batch_size: int) -> Any:
    legacy, cache_type = _cache_as_legacy(cache)
    expanded_layers: list[Any] = []
    for layer in legacy:
        key, value = layer
        expanded_pair = (_expand_batch(key, count), _expand_batch(value, count))
        expanded_layers.append(
            list(expanded_pair) if isinstance(layer, list) else expanded_pair
        )
    if cache_type is None:
        expanded: Any = (
            tuple(expanded_layers) if isinstance(cache, tuple) else expanded_layers
        )
    else:
        factory = getattr(cache_type, "from_legacy_cache")
        legacy_container = (
            tuple(expanded_layers) if isinstance(legacy, tuple) else expanded_layers
        )
        expanded = factory(legacy_container)
        if type(expanded) is not cache_type:
            raise TypeError(
                "from_legacy_cache() changed KV cache type from "
                f"{_qualified_type(cache)} to {_qualified_type(expanded)}"
            )
    first_key = expanded_layers[0][0]
    _validate_cache(
        expanded,
        batch_size=batch_size * count,
        device=first_key.device,
    )
    return expanded


def _make_att_2d_masks(
    prefix_pad_masks: torch.Tensor, prefix_att_masks: torch.Tensor
) -> torch.Tensor:
    cumulative = torch.cumsum(prefix_att_masks, dim=1)
    attention = cumulative[:, None, :] <= cumulative[:, :, None]
    padding = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
    return attention & padding.to(dtype=torch.bool)


class OpenPIParallelVerifier:
    """Bind one loaded OpenPI checkpoint to safe ``B*K`` draft verification."""

    def __init__(
        self,
        target: VerifiedOpenPITarget,
        *,
        velocity_target: VelocityTarget | None = None,
    ) -> None:
        if not isinstance(target, VerifiedOpenPITarget):
            raise TypeError("target must be a VerifiedOpenPITarget")
        target.validate_runtime_model()
        self.verified_target = target
        self.model = target.model
        self.target_contract = target.contract
        self.loaded_checkpoint_id = target.contract.checkpoint_id
        self._validate_model_config()
        self._velocity_target = velocity_target
        self._owner_token = object()
        self._active_episode_id: str | None = None
        self._generation = 0
        self._current_bundle: PrefixBundle | None = None

    def _validate_model_config(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None:
            raise ConfigurationError("OpenPI target must expose model.config")
        expected = self.target_contract
        for field_name, expected_value in (
            ("action_horizon", expected.action_horizon),
            ("action_dim", expected.model_action_dim),
        ):
            actual = getattr(config, field_name, None)
            if actual is None or int(actual) != expected_value:
                raise ConfigurationError(
                    f"model.config.{field_name}={actual!r} does not match "
                    f"target contract {expected_value}"
                )
        action_env_dim = getattr(config, "action_env_dim", None)
        if action_env_dim is not None and int(action_env_dim) != len(
            expected.env_action_indices
        ):
            raise ConfigurationError(
                "model.config.action_env_dim does not match explicit env_action_indices"
            )
        config_name = getattr(config, "config_name", None)
        if config_name is not None and str(config_name) != expected.config_name:
            raise ConfigurationError(
                "model.config.config_name does not match target contract"
            )
        pi05 = getattr(config, "pi05", None)
        if pi05 is not None:
            family = "pi0.5" if bool(pi05) else "pi0"
            if family != expected.policy_family:
                raise ConfigurationError(
                    "model.config.pi05 does not match target policy_family"
                )

    def begin_episode(self, episode_id: str) -> None:
        """Claim an episode; reset is required before another begin call."""

        episode_id = _ensure_nonempty(episode_id, "episode_id")
        if self._active_episode_id is not None:
            raise RuntimeError(
                "an OpenPI verifier episode is already active; reset it first"
            )
        self._generation += 1
        self._active_episode_id = episode_id
        self._current_bundle = None

    def reset_episode(self, episode_id: str | None = None) -> None:
        """Invalidate the current prefix/KV cache at an episode boundary."""

        if episode_id is not None:
            episode_id = _ensure_nonempty(episode_id, "episode_id")
            if episode_id != self._active_episode_id:
                raise RuntimeError("cannot reset a prefix owned by another episode")
        self._generation += 1
        self._active_episode_id = None
        self._current_bundle = None

    def _assert_active_episode(self, episode_id: str) -> None:
        if self._active_episode_id is None:
            raise RuntimeError(
                "begin_episode() must be called before prefix preparation"
            )
        if episode_id != self._active_episode_id:
            raise RuntimeError(
                "prefix episode_id does not own the active verifier cache"
            )

    def prepare_prefix(self, observation: Any, *, episode_id: str) -> PrefixBundle:
        """Run one OpenPI encoder and one contextual VLM prefill."""

        episode_id = _ensure_nonempty(episode_id, "episode_id")
        self._assert_active_episode(episode_id)
        preprocess = getattr(self.model, "_preprocess_observation", None)
        embed_prefix = getattr(self.model, "embed_prefix", None)
        if not callable(preprocess) or not callable(embed_prefix):
            raise ConfigurationError(
                "OpenPI target must expose _preprocess_observation and embed_prefix"
            )
        with torch.no_grad():
            prepared = preprocess(observation, train=False)
            if not isinstance(prepared, (tuple, list)) or len(prepared) != 5:
                raise TypeError(
                    "_preprocess_observation must return images, image masks, "
                    "language tokens, language masks, and normalized state"
                )
            images, image_masks, language_tokens, language_masks, state = prepared
            prefix = embed_prefix(
                images,
                image_masks,
                language_tokens,
                language_masks,
            )
            if not isinstance(prefix, (tuple, list)) or len(prefix) != 3:
                raise TypeError(
                    "embed_prefix must return embeddings, padding masks, and attention masks"
                )
            prefix_embs, prefix_pad_masks, prefix_att_masks = prefix
            bundle = self._contextualize_prefix(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                state,
                episode_id=episode_id,
            )
        self._current_bundle = bundle
        return bundle

    def _contextualize_prefix(
        self,
        prefix_embs: Any,
        prefix_pad_masks: Any,
        prefix_att_masks: Any,
        normalized_state: Any,
        *,
        episode_id: str,
    ) -> PrefixBundle:
        state = _ensure_floating_finite(normalized_state, "normalized_state", ndim=2)
        expected_state_shape = (int(state.shape[0]), self.target_contract.state_dim)
        if tuple(state.shape) != expected_state_shape:
            raise ShapeMismatchError(
                f"normalized_state must have shape {expected_state_shape}"
            )
        if int(state.shape[0]) <= 0:
            raise ShapeMismatchError("normalized_state batch must be positive")
        embeddings = _ensure_floating_finite(prefix_embs, "prefix_embs", ndim=3)
        batch_size, prefix_length = (int(value) for value in embeddings.shape[:2])
        if prefix_length <= 0 or int(embeddings.shape[2]) <= 0:
            raise ShapeMismatchError(
                "prefix token length and embedding width must be positive"
            )
        if batch_size != int(state.shape[0]):
            raise ShapeMismatchError(
                "prefix embeddings and normalized state batches differ"
            )
        mask_shape = (batch_size, prefix_length)
        pad_masks = _ensure_mask(prefix_pad_masks, "prefix_pad_masks", shape=mask_shape)
        att_masks = _ensure_mask(prefix_att_masks, "prefix_att_masks", shape=mask_shape)
        for name, tensor in (
            ("prefix_embs", embeddings),
            ("prefix_pad_masks", pad_masks),
            ("prefix_att_masks", att_masks),
        ):
            if tensor.device != state.device:
                raise ValueError(
                    f"{name} and normalized_state are on different devices"
                )
        if bool((pad_masks.to(dtype=torch.int64).sum(dim=1) == 0).any().item()):
            raise ShapeMismatchError(
                "each prefix must contain at least one valid token"
            )

        attention_2d = _make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        prepare_4d = getattr(self.model, "_prepare_attention_masks_4d", None)
        if not callable(prepare_4d):
            raise ConfigurationError(
                "OpenPI target must expose _prepare_attention_masks_4d"
            )
        attention_4d = prepare_4d(attention_2d)
        forward_owner = getattr(self.model, "paligemma_with_expert", None)
        forward = getattr(forward_owner, "forward", None)
        if not callable(forward):
            raise ConfigurationError(
                "OpenPI target must expose paligemma_with_expert.forward"
            )
        language_model = getattr(
            getattr(forward_owner, "paligemma", None), "language_model", None
        )
        language_config = getattr(language_model, "config", None)
        if language_config is not None and hasattr(
            language_config, "_attn_implementation"
        ):
            language_config._attn_implementation = "eager"  # noqa: SLF001
        forward_result = forward(
            attention_mask=attention_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        if not isinstance(forward_result, (tuple, list)) or len(forward_result) != 2:
            raise TypeError("OpenPI prefill forward must return outputs and KV cache")
        outputs, past_key_values = forward_result
        if isinstance(outputs, (tuple, list)):
            if not outputs:
                raise TypeError("OpenPI prefill outputs must contain prefix output")
            contextual = outputs[0]
        else:
            contextual = outputs
        contextual = _ensure_floating_finite(
            contextual, "contextual_prefix_output", ndim=3
        )
        if contextual.shape[:2] != embeddings.shape[:2]:
            raise ShapeMismatchError(
                "contextual prefix output must align with raw prefix tokens"
            )
        if contextual.device != state.device:
            raise ValueError(
                "contextual prefix output and normalized state are on different devices"
            )
        _validate_cache(
            past_key_values,
            batch_size=batch_size,
            device=state.device,
        )
        return PrefixBundle(
            prefix_embs=embeddings,
            prefix_pad_masks=pad_masks,
            prefix_att_masks=att_masks,
            normalized_state=state,
            contextual_prefix_output=contextual,
            past_key_values=past_key_values,
            episode_id=episode_id,
            target_contract=self.target_contract,
            _owner_token=self._owner_token,
            _generation=self._generation,
        )

    def _validate_bundle(self, bundle: PrefixBundle) -> None:
        if not isinstance(bundle, PrefixBundle):
            raise TypeError("prefix_bundle must be a PrefixBundle")
        if bundle._owner_token is not self._owner_token:
            raise RuntimeError("prefix bundle belongs to a different verifier")
        if bundle._generation != self._generation or bundle is not self._current_bundle:
            raise RuntimeError("prefix bundle is stale or was invalidated by reset")
        if bundle.episode_id != self._active_episode_id:
            raise RuntimeError("prefix bundle belongs to a different episode")
        if (
            bundle.target_contract.compatibility_key
            != self.target_contract.compatibility_key
        ):
            raise ConfigurationError("prefix bundle target contract changed")

    def _call_velocity(
        self,
        state: torch.Tensor,
        pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        if self._velocity_target is not None:
            output = self._velocity_target(
                state, pad_masks, past_key_values, x_t, timesteps
            )
            path = "explicit_velocity_target"
        else:
            get_velocity = getattr(self.model, "get_velocity", None)
            denoise_step = getattr(self.model, "denoise_step", None)
            get_suffix_out = getattr(self.model, "get_suffix_out", None)
            if callable(get_velocity):
                # RLinf's current OpenPI target exposes the native order below
                # (see OpenPINativeRTCAdapter._velocity).  The release/v0.2
                # get_suffix_out fallback intentionally retains its older
                # state/mask/cache/x/t order in the branch below.
                output = get_velocity(
                    state,
                    x_t,
                    timesteps,
                    pad_masks,
                    past_key_values,
                )
                path = "model.get_velocity"
            elif callable(denoise_step):
                output = denoise_step(state, pad_masks, past_key_values, x_t, timesteps)
                path = "model.denoise_step"
            elif callable(get_suffix_out) and callable(
                getattr(self.model, "action_out_proj", None)
            ):
                suffix = get_suffix_out(
                    state, pad_masks, past_key_values, x_t, timesteps
                )
                output = self.model.action_out_proj(suffix)
                path = "model.get_suffix_out+action_out_proj"
            else:
                raise ConfigurationError(
                    "OpenPI target exposes no supported velocity/denoiser method"
                )
        if isinstance(output, (tuple, list)):
            if not output:
                raise TypeError("OpenPI velocity result is empty")
            output = output[0]
        velocity = _ensure_floating_finite(output, "predicted_velocity", ndim=3)
        if tuple(velocity.shape) != tuple(x_t.shape):
            raise ShapeMismatchError(
                "OpenPI velocity must match flattened [B*K, H, A_model]"
            )
        if velocity.device != x_t.device:
            raise ValueError(
                "OpenPI velocity and interpolation are on different devices"
            )
        return velocity, path

    def verify(
        self,
        prefix_bundle: PrefixBundle,
        draft_prediction: VerifiedDraftPrediction,
        noise: torch.Tensor,
        verification_times: torch.Tensor,
    ) -> OpenPIParallelVerification:
        """Verify ``K`` flow points in exactly one target denoiser call."""

        self._validate_bundle(prefix_bundle)
        if not isinstance(draft_prediction, VerifiedDraftPrediction):
            raise TypeError(
                "draft_prediction must be a VerifiedDraftPrediction issued by a "
                "verified production Draft loader"
            )
        if draft_prediction._verification_proof is not _VERIFIED_DRAFT_PREDICTION_PROOF:
            raise ConfigurationError("Draft prediction verification proof is invalid")
        if draft_prediction._target_token is not self.verified_target._target_token:
            raise ConfigurationError(
                "Draft prediction was issued for a different loaded OpenPI target"
            )
        self.target_contract.require_compatible_draft(draft_prediction.target_contract)
        draft = _ensure_floating_finite(
            draft_prediction.actions_model, "draft_model", ndim=3
        )
        shared_noise = _ensure_floating_finite(noise, "noise", ndim=3)
        times = _ensure_floating_finite(
            verification_times, "verification_times", ndim=1
        )
        expected_shape = (
            prefix_bundle.batch_size,
            self.target_contract.action_horizon,
            self.target_contract.model_action_dim,
        )
        if tuple(draft.shape) != expected_shape:
            raise ShapeMismatchError(f"draft_model must have shape {expected_shape}")
        if tuple(shared_noise.shape) != expected_shape:
            raise ShapeMismatchError(f"noise must have shape {expected_shape}")
        if times.numel() <= 0:
            raise ShapeMismatchError("verification_times must contain at least one K")
        for name, tensor in (
            ("draft_model", draft),
            ("noise", shared_noise),
            ("verification_times", times),
        ):
            if tensor.device != prefix_bundle.device:
                raise ValueError(f"{name} and prefix bundle are on different devices")
        if shared_noise.dtype != draft.dtype or times.dtype != draft.dtype:
            raise TypeError(
                "draft_model, noise, and verification_times must use one exact dtype"
            )
        if bool(((times <= 0) | (times > 1)).any().item()):
            raise ValueError("verification_times must lie in (0, 1]")

        batch_size, horizon, action_dim = expected_shape
        verification_count = int(times.numel())
        times_bkhd = times.view(1, verification_count, 1, 1)
        interpolation = (
            times_bkhd * shared_noise[:, None] + (1 - times_bkhd) * draft[:, None]
        )
        interpolation_flat = interpolation.reshape(
            batch_size * verification_count, horizon, action_dim
        )
        timestep_flat = (
            times.view(1, verification_count)
            .expand(batch_size, verification_count)
            .reshape(batch_size * verification_count)
        )
        expanded_state = _expand_batch(
            prefix_bundle.normalized_state, verification_count
        )
        expanded_pad_masks = _expand_batch(
            prefix_bundle.prefix_pad_masks, verification_count
        )
        # Attention masks are not consumed after prefill by current OpenPI
        # denoisers, but expanding and validating them here keeps the complete
        # B*K conditioning contract explicit for future target adapters.
        expanded_att_masks = _expand_batch(
            prefix_bundle.prefix_att_masks, verification_count
        )
        if expanded_att_masks.shape != expanded_pad_masks.shape:
            raise ShapeMismatchError("expanded OpenPI prefix masks differ")
        expanded_cache = _expand_kv_cache(
            prefix_bundle.past_key_values,
            verification_count,
            batch_size=batch_size,
        )
        with torch.no_grad():
            velocity_flat, velocity_path = self._call_velocity(
                expanded_state,
                expanded_pad_masks,
                expanded_cache,
                interpolation_flat,
                timestep_flat,
            )
        velocity = velocity_flat.reshape(
            batch_size, verification_count, horizon, action_dim
        )
        x0_hat = interpolation - times_bkhd * velocity
        if not bool(torch.isfinite(x0_hat).all().item()):
            raise ValueError("x0_hat contains NaN or infinity")
        env_indices = torch.tensor(
            self.target_contract.env_action_indices,
            dtype=torch.long,
            device=draft.device,
        )
        draft_env = draft.index_select(-1, env_indices)
        noise_env = shared_noise.index_select(-1, env_indices)
        velocity_env = velocity.index_select(-1, env_indices)
        x0_hat_env = x0_hat.index_select(-1, env_indices)
        return OpenPIParallelVerification(
            draft_model=draft,
            shared_noise=shared_noise,
            verification_times=times,
            interpolation_model=interpolation,
            predicted_velocity_model=velocity,
            x0_hat_model=x0_hat,
            draft_env=draft_env,
            shared_noise_env=noise_env,
            predicted_velocity_env=velocity_env,
            x0_hat_env=x0_hat_env,
            env_action_indices=self.target_contract.env_action_indices,
            velocity_path=velocity_path,
            kv_cache_type=prefix_bundle.kv_cache_type,
            expanded_kv_cache_type=_qualified_type(expanded_cache),
            episode_id=prefix_bundle.episode_id,
            target_contract=self.target_contract,
            draft_checkpoint_id=draft_prediction.draft_checkpoint_id,
            draft_checkpoint_sha256=draft_prediction.draft_checkpoint_sha256,
        )


def triton_postprocess_batch(
    verification: OpenPIParallelVerification,
    config: Any,
    *,
    previous_gripper_values: Sequence[float | None] | None = None,
    backend: Literal["auto", "torch", "triton"] = "auto",
) -> tuple[Any, ...]:
    """Optionally apply the existing post-processing to each batch item.

    This helper does not load or bless any draft checkpoint.  Compatibility
    has already been enforced by :meth:`OpenPIParallelVerifier.verify`, while
    this function only performs deterministic environment-dimension
    interpolation/error reduction and stitching.
    """

    from .triton_speculative import (
        TritonSpeculativeConfig,
        speculative_verify_and_stitch,
    )

    if not isinstance(verification, OpenPIParallelVerification):
        raise TypeError("verification must be an OpenPIParallelVerification")
    if not isinstance(config, TritonSpeculativeConfig):
        raise TypeError("config must be a TritonSpeculativeConfig")
    batch_size = int(verification.draft_env.shape[0])
    if previous_gripper_values is None:
        gripper_values: tuple[float | None, ...] = (None,) * batch_size
    else:
        gripper_values = tuple(previous_gripper_values)
        if len(gripper_values) != batch_size:
            raise ShapeMismatchError(
                "previous_gripper_values must contain one value per batch item"
            )
    results = []
    for batch_index in range(batch_size):
        results.append(
            speculative_verify_and_stitch(
                verification.draft_env[batch_index],
                verification.shared_noise_env[batch_index],
                verification.predicted_velocity_env[batch_index],
                verification.verification_times,
                config,
                previous_gripper_value=gripper_values[batch_index],
                backend=backend,
            )
        )
    return tuple(results)


__all__ = [
    "OpenPIParallelVerification",
    "OpenPIParallelVerifier",
    "OpenPITargetContract",
    "PrefixBundle",
    "triton_postprocess_batch",
]
