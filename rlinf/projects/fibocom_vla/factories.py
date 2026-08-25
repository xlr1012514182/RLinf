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

"""Policy factories used by the command-line runtime."""

from __future__ import annotations

import operator
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import StackConfig
from .contracts import ChunkPolicy
from .errors import ConfigurationError


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required")
    return value


def _optional_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def _positive_int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _required_file_bytes(name: str) -> bytes:
    path = Path(_required_env(name)).expanduser()
    if not path.is_absolute():
        raise ConfigurationError(f"{name} must be an absolute file path")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            f"failed to read {name}={str(path)!r}: {error}"
        ) from error


def _verification_times_from_env() -> tuple[float, ...]:
    name = "FIBOCOM_PI05_SPECULATIVE_VERIFICATION_TIMES"
    raw = os.environ.get(name, "0.10,0.05")
    fields = tuple(field.strip() for field in raw.split(","))
    if not fields or any(not field for field in fields):
        raise ConfigurationError(f"{name} must be a comma-separated float sequence")
    try:
        return tuple(float(field) for field in fields)
    except ValueError as error:
        raise ConfigurationError(
            f"{name} must be a comma-separated float sequence"
        ) from error


def _camera_mapping(config: StackConfig) -> tuple[str, tuple[str, ...]]:
    declared = {camera.name for camera in config.cameras}
    required = tuple(config.robot.action_adapter.required_cameras)
    main = os.environ.get("FIBOCOM_MAIN_CAMERA", "main").strip()
    if main not in declared:
        raise ConfigurationError(
            f"FIBOCOM_MAIN_CAMERA={main!r} is not a declared camera"
        )
    explicit_wrist = os.environ.get("FIBOCOM_WRIST_CAMERA")
    wrist = explicit_wrist.strip() if explicit_wrist else None
    if wrist is None and "wrist" in required:
        wrist = "wrist"
    if wrist is not None and wrist not in declared:
        raise ConfigurationError(
            f"FIBOCOM_WRIST_CAMERA={wrist!r} is not a declared camera"
        )
    mapped = {main} | ({wrist} if wrist is not None else set())
    missing = sorted(set(required) - mapped)
    if missing:
        raise ConfigurationError(
            "OpenPI adapter cannot map required policy cameras: " + ", ".join(missing)
        )
    return main, (() if wrist is None else (wrist,))


def _asset_bound_model_inputs(
    config: StackConfig,
) -> tuple[str, str, tuple[str, ...], Any | None]:
    """Resolve an optional source-locked checkpoint before model construction."""

    model_path = _required_env("FIBOCOM_PI05_MODEL_PATH")
    manifest_path = os.environ.get("FIBOCOM_PI05_ASSET_MANIFEST", "").strip()
    if not manifest_path:
        _required_env("FIBOCOM_PI05_CONFIG_NAME")
        main_camera, wrist_cameras = _camera_mapping(config)
        return model_path, main_camera, wrist_cameras, None

    checkpoint_root = Path(model_path).expanduser()
    if not checkpoint_root.is_absolute():
        raise ConfigurationError(
            "FIBOCOM_PI05_MODEL_PATH must be absolute when an asset manifest is used"
        )
    from .assets import (
        load_and_verify_checkpoint_assets,
        require_verified_checkpoint_assets,
    )

    repository_root = Path(__file__).resolve().parents[3]
    verified = require_verified_checkpoint_assets(
        load_and_verify_checkpoint_assets(
            manifest_path,
            checkpoint_root,
            repository_root=repository_root,
        )
    )
    verified.manifest.validate_stack(config)
    runtime = verified.manifest.runtime
    explicit_config = os.environ.get("FIBOCOM_PI05_CONFIG_NAME", "").strip()
    if explicit_config and explicit_config != runtime.config_name:
        raise ConfigurationError(
            "FIBOCOM_PI05_CONFIG_NAME conflicts with the source-locked manifest: "
            f"{explicit_config!r} != {runtime.config_name!r}"
        )
    explicit_main = os.environ.get("FIBOCOM_MAIN_CAMERA", "").strip()
    if explicit_main and explicit_main != runtime.cameras.main:
        raise ConfigurationError(
            "FIBOCOM_MAIN_CAMERA conflicts with the checkpoint camera contract"
        )
    explicit_wrist = os.environ.get("FIBOCOM_WRIST_CAMERA", "").strip()
    if explicit_wrist:
        raise ConfigurationError(
            "FIBOCOM_WRIST_CAMERA cannot override a manifest-bound ordered camera set"
        )
    required = set(config.robot.action_adapter.required_cameras)
    mapped = set(runtime.cameras.observation_keys)
    if not required.issubset(mapped):
        raise ConfigurationError(
            "checkpoint camera contract does not cover action-adapter cameras: "
            + ", ".join(sorted(required - mapped))
        )
    return (
        str(verified.root),
        runtime.cameras.main,
        runtime.cameras.wrists,
        verified,
    )


def _model_config_int(model: Any, name: str) -> int:
    model_config = getattr(model, "config", None)
    if model_config is None or not hasattr(model_config, name):
        raise ConfigurationError(f"loaded OpenPI model config lacks {name!r}")
    value = getattr(model_config, name)
    if isinstance(value, bool):
        raise ConfigurationError(f"OpenPI model config {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ConfigurationError(
            f"OpenPI model config {name} must be an integer, got {value!r}"
        ) from error


def _registered_value(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise ConfigurationError(f"registered OpenPI config lacks {name!r}")
    return getattr(value, name)


def _validate_registered_openpi_contract(registered: Any, runtime: Any) -> None:
    """Bind the manifest to the complete registered preprocessing contract."""

    model_config = _registered_value(registered, "model")
    data_factory = _registered_value(registered, "data")
    checks = {
        "name": (_registered_value(registered, "name"), runtime.config_name),
        "model.pi05": (_registered_value(model_config, "pi05"), True),
        "model.discrete_state_input": (
            _registered_value(model_config, "discrete_state_input"),
            runtime.discrete_state_input,
        ),
        "model.action_horizon": (
            _registered_value(model_config, "action_horizon"),
            runtime.action_horizon,
        ),
        "model.action_dim": (
            _registered_value(model_config, "action_dim"),
            runtime.model_action_dim,
        ),
        "model.max_token_len": (
            _registered_value(model_config, "max_token_len"),
            runtime.max_token_length,
        ),
        "data.repo_id": (
            _registered_value(data_factory, "repo_id"),
            runtime.asset_id,
        ),
        "data.adapt_to_pi": (
            _registered_value(data_factory, "adapt_to_pi"),
            runtime.adapt_to_pi,
        ),
        "data.extra_delta_transform": (
            _registered_value(data_factory, "extra_delta_transform"),
            runtime.extra_delta_transform,
        ),
    }
    assets = _registered_value(data_factory, "assets")
    configured_asset_id = getattr(assets, "asset_id", None) or _registered_value(
        data_factory, "repo_id"
    )
    checks["data.asset_id"] = (configured_asset_id, runtime.asset_id)
    mismatches = [
        f"{name}: registered={observed!r}, manifest={expected!r}"
        for name, (observed, expected) in checks.items()
        if observed != expected
    ]
    if runtime.normalization != "quantile_q01_q99":
        mismatches.append("normalization: registered pi0.5 requires quantile_q01_q99")
    if mismatches:
        raise ConfigurationError(
            "registered OpenPI config conflicts with the checkpoint manifest: "
            + "; ".join(mismatches)
        )


def _validate_openpi_load_report(model: Any, verified_assets: Any) -> None:
    report = getattr(model, "_rlinf_checkpoint_load_report", None)
    if not isinstance(report, Mapping):
        raise ConfigurationError(
            "loaded OpenPI model lacks a checkpoint load compatibility report"
        )
    if report.get("source_kind") != "safetensors_shards":
        raise ConfigurationError(
            "manifest-bound OpenPI model did not load from verified safetensors shards"
        )
    runtime = verified_assets.manifest.runtime
    report_checks = {
        "data_asset_id": runtime.asset_id,
        "use_quantile_norm": True,
    }
    report_mismatches = [
        f"{name}: loaded={report.get(name)!r}, manifest={expected!r}"
        for name, expected in report_checks.items()
        if report.get(name) != expected
    ]
    if report_mismatches:
        raise ConfigurationError(
            "loaded OpenPI preprocessing conflicts with the manifest: "
            + "; ".join(report_mismatches)
        )
    from .assets import checkpoint_load_key_classes

    try:
        missing, unexpected, ignored_auxiliary, unresolved = (
            checkpoint_load_key_classes(dict(report))
        )
    except ValueError as error:
        raise ConfigurationError(
            f"OpenPI checkpoint load report is invalid: {error}"
        ) from error
    if missing or unresolved:
        raise ConfigurationError(
            "OpenPI checkpoint main state_dict is not exact-key compatible: "
            f"missing={missing}, raw_unexpected={unexpected}, "
            f"ignored_auxiliary={ignored_auxiliary}, unresolved={unresolved}"
        )
    observed_paths = {
        Path(path).expanduser().resolve() for path in report.get("selected_paths", ())
    }
    expected_paths = {path.resolve() for path in verified_assets.weight_paths}
    if observed_paths != expected_paths:
        raise ConfigurationError(
            "OpenPI loader selected files differ from the verified manifest shards"
        )


def _validate_openpi_shape_contract(
    model: Any,
    config: StackConfig,
    runtime: Any | None = None,
) -> None:
    """Reject checkpoint/config shapes that cannot produce the declared chunk."""

    expected_horizon = config.residual_rl.action_horizon
    expected_action_dim = config.residual_rl.action_dim
    model_horizon = _model_config_int(model, "action_horizon")
    output_chunk = _model_config_int(model, "action_chunk")
    output_action_dim = _model_config_int(model, "action_env_dim")
    padded_action_dim = _model_config_int(model, "action_dim")

    if model_horizon != expected_horizon:
        raise ConfigurationError(
            "OpenPI checkpoint/config action_horizon mismatch: loaded "
            f"{model_horizon}, but the stack requires {expected_horizon}. RLinf "
            "action_chunk only truncates model output and cannot extend a shorter "
            "horizon. Select a registered config/checkpoint trained for the exact "
            "stack horizon; do not silently override stock pi05_libero."
        )
    if output_chunk != expected_horizon:
        raise ConfigurationError(
            "OpenPI action_chunk mismatch after loading: "
            f"loaded {output_chunk}, expected {expected_horizon}"
        )
    if output_action_dim != expected_action_dim:
        raise ConfigurationError(
            "OpenPI action_env_dim mismatch after loading: "
            f"loaded {output_action_dim}, expected {expected_action_dim}"
        )
    if runtime is None and padded_action_dim < expected_action_dim:
        raise ConfigurationError(
            "OpenPI padded action_dim is smaller than the requested environment "
            f"action dimension: {padded_action_dim} < {expected_action_dim}"
        )
    if runtime is not None:
        exact_checks = {
            "config_name": runtime.config_name,
            "pi05": True,
            "discrete_state_input": runtime.discrete_state_input,
            "action_dim": runtime.model_action_dim,
            "action_horizon": runtime.action_horizon,
            "max_token_len": runtime.max_token_length,
            "action_env_dim": runtime.environment_action_dim,
            "action_chunk": runtime.action_horizon,
            "num_images_in_input": len(runtime.cameras.observation_keys),
        }
        model_config = getattr(model, "config", None)
        mismatches = []
        for name, expected in exact_checks.items():
            if model_config is None or not hasattr(model_config, name):
                mismatches.append(f"{name}: missing")
                continue
            observed = getattr(model_config, name)
            if observed != expected:
                mismatches.append(f"{name}: loaded={observed!r}, manifest={expected!r}")
        if mismatches:
            raise ConfigurationError(
                "loaded OpenPI model config conflicts with the manifest: "
                + "; ".join(mismatches)
            )


def create_openpi_policy_from_env(config: StackConfig):
    """Load an RLinf π0.5 checkpoint using explicit environment variables.

    Required:
        ``FIBOCOM_PI05_MODEL_PATH`` points to an OpenPI checkpoint containing
        the normalization statistics expected by RLinf.

    ``FIBOCOM_PI05_CONFIG_NAME`` is required only without an asset manifest;
    manifest-bound loads derive it from the verified runtime contract.
    ``FIBOCOM_PI05_DEVICE`` defaults to ``cuda``. Camera keys are checked
    against the action-adapter contract.

    When residual/speculative layers are enabled, their assets are mandatory as
    well. Production speculative inference accepts only a manifest-bound main
    target plus an evaluated, Ed25519-approved Draft artifact. Missing
    integration never silently degrades to a naked base policy.
    """

    if config.speculative.enabled and config.residual_rl.enabled:
        raise ConfigurationError(
            "residual_rl and speculative cannot be enabled together: the signed "
            "Draft verifier is bound to the exact frozen main checkpoint, not a "
            "residual-modified action policy"
        )
    if (
        config.speculative.enabled
        and not os.environ.get("FIBOCOM_PI05_ASSET_MANIFEST", "").strip()
    ):
        raise ConfigurationError(
            "FIBOCOM_PI05_ASSET_MANIFEST is required for production speculative "
            "inference"
        )

    model_path, main_camera, wrist_cameras, verified_assets = _asset_bound_model_inputs(
        config
    )
    device = os.environ.get("FIBOCOM_PI05_DEVICE", "cuda")
    config_name = (
        _required_env("FIBOCOM_PI05_CONFIG_NAME")
        if verified_assets is None
        else verified_assets.manifest.runtime.config_name
    )
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi import get_model

    from .openpi_adapter import OpenPiAdapterConfig, RLinfOpenPiChunkPolicy

    openpi_config = {
        "config_name": config_name,
        "action_chunk": config.residual_rl.action_horizon,
        "action_env_dim": config.residual_rl.action_dim,
        "num_steps": int(os.environ.get("FIBOCOM_PI05_DENOISE_STEPS", "5")),
        "train_expert_only": True,
        "add_value_head": False,
        "noise_method": os.environ.get("FIBOCOM_PI05_NOISE_METHOD", "flow_sde"),
    }
    if verified_assets is not None:
        runtime = verified_assets.manifest.runtime
        # The source-locked registered config is independently checked below;
        # this explicit value makes the realized runtime geometry auditable.
        openpi_config["action_horizon"] = runtime.action_horizon
        openpi_config["num_images_in_input"] = len(runtime.cameras.observation_keys)
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        registered = get_openpi_config(config_name, model_path=model_path)
        _validate_registered_openpi_contract(registered, runtime)
    model_config = OmegaConf.create({"model_path": model_path, "openpi": openpi_config})
    model = get_model(model_config)
    runtime = None if verified_assets is None else verified_assets.manifest.runtime
    _validate_openpi_shape_contract(model, config, runtime)
    if verified_assets is not None:
        _validate_openpi_load_report(model, verified_assets)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual_graph_patch = None
    if _optional_bool_env("FIBOCOM_PI05_CUDA_GRAPH_VISUAL"):
        if not str(device).lower().startswith("cuda"):
            raise ConfigurationError(
                "FIBOCOM_PI05_CUDA_GRAPH_VISUAL requires a CUDA model device"
            )
        from .inference.openpi_cuda_graph import (
            OpenPIVisualExecutionContext,
            install_rlinf_openpi_visual_graph,
        )

        # The captured component is deliberately limited to PaliGemma's
        # shape-stable vision tower + multimodal projector. Native RTC needs
        # autograd/VJP and speculative verification has dynamic control flow;
        # both therefore force the patched call site onto its eager route.
        visual_graph_patch = install_rlinf_openpi_visual_graph(
            model,
            cache_capacity=_positive_int_env(
                "FIBOCOM_PI05_CUDA_GRAPH_CACHE_CAPACITY", default=4
            ),
            warmup_iterations=_positive_int_env(
                "FIBOCOM_PI05_CUDA_GRAPH_WARMUP", default=3
            ),
            default_context=OpenPIVisualExecutionContext(
                rtc=config.rtc.enabled,
                dynamic_branch=config.speculative.enabled,
            ),
        )
    verified_openpi_target = None
    if verified_assets is not None:
        from .inference.openpi_speculative import VerifiedOpenPITarget

        verified_openpi_target = VerifiedOpenPITarget.from_verified_assets(
            model,
            verified_assets,
        )
    base_policy = RLinfOpenPiChunkPolicy(
        model,
        OpenPiAdapterConfig(
            main_camera=main_camera,
            wrist_cameras=wrist_cameras,
            period_s=1.0 / config.robot.control_hz,
            expected_state_names=(() if runtime is None else runtime.state_semantics),
        ),
    )
    if visual_graph_patch is not None:
        setattr(base_policy, "fibocom_visual_graph_patch", visual_graph_patch)
    if verified_openpi_target is not None:
        setattr(
            base_policy,
            "fibocom_verified_openpi_target",
            verified_openpi_target,
        )
    policy: ChunkPolicy = base_policy
    layers = ["rlinf_openpi_pi05"]
    if visual_graph_patch is not None:
        layers.append("cuda_graph_visual_static_eager_guard")

    if config.residual_rl.enabled:
        checkpoint = _required_env("FIBOCOM_RESIDUAL_CHECKPOINT")
        from .residual_policy import ResidualChunkPolicy

        policy = ResidualChunkPolicy.from_checkpoint(
            base_policy,
            config.residual_rl,
            checkpoint,
            device=device,
            stochastic=False,
        )
        layers.append("bounded_residual_actor")

    if config.speculative.enabled:
        if verified_openpi_target is None:
            raise ConfigurationError(
                "production speculative inference requires a verified OpenPI target"
            )
        draft_manifest = _required_env("FIBOCOM_PI05_DRAFT_MANIFEST")
        draft_approval = _required_env("FIBOCOM_PI05_DRAFT_APPROVAL")
        draft_evaluation = _required_env("FIBOCOM_PI05_DRAFT_EVALUATION")
        draft_public_key = _required_file_bytes("FIBOCOM_PI05_DRAFT_PUBLIC_KEY")
        draft_signing_key_id = _required_env("FIBOCOM_PI05_DRAFT_SIGNING_KEY_ID")
        draft_dtype = os.environ.get("FIBOCOM_PI05_DRAFT_DTYPE", "bfloat16").strip()
        if draft_dtype not in {"float32", "bfloat16"}:
            raise ConfigurationError(
                "FIBOCOM_PI05_DRAFT_DTYPE must be 'float32' or 'bfloat16'"
            )
        verification_backend = os.environ.get(
            "FIBOCOM_PI05_SPECULATIVE_BACKEND", "auto"
        ).strip()
        if verification_backend not in {"auto", "torch", "triton"}:
            raise ConfigurationError(
                "FIBOCOM_PI05_SPECULATIVE_BACKEND must be 'auto', 'torch', or 'triton'"
            )
        episode_id = os.environ.get(
            "FIBOCOM_PI05_SPECULATIVE_EPISODE_ID", "factory-episode-0"
        ).strip()
        if not episode_id:
            raise ConfigurationError(
                "FIBOCOM_PI05_SPECULATIVE_EPISODE_ID must be non-empty"
            )
        from .inference.openpi_production_speculative import (
            OpenPIProductionSpeculativePolicy,
        )
        from .inference.pi05_draft import (
            Pi05DraftTargetContract,
            load_production_pi05_draft,
        )

        expected_draft_target = Pi05DraftTargetContract.from_verified_openpi(
            verified_openpi_target,
            draft_parameter_dtype=draft_dtype,
        )
        draft = load_production_pi05_draft(
            candidate_manifest_path=draft_manifest,
            approval_path=draft_approval,
            evaluation_report_path=draft_evaluation,
            expected_target=expected_draft_target,
            trusted_public_key_raw=draft_public_key,
            trusted_signing_key_id=draft_signing_key_id,
            device=device,
        )
        policy = OpenPIProductionSpeculativePolicy(
            verified_openpi_target,
            draft,
            base_policy,
            config.speculative,
            episode_id=episode_id,
            verification_times=_verification_times_from_env(),
            backend=verification_backend,
        )
        layers.append("openpi_production_signed_draft_parallel_verify")

    setattr(policy, "fibocom_stack_layers", tuple(layers))
    setattr(policy, "fibocom_native_rtc_model_lineage", policy is base_policy)
    if verified_openpi_target is not None:
        setattr(
            policy,
            "fibocom_verified_openpi_target",
            verified_openpi_target,
        )
    if visual_graph_patch is not None:
        setattr(policy, "fibocom_visual_graph_patch", visual_graph_patch)
    if verified_assets is not None:
        setattr(
            policy,
            "fibocom_checkpoint_identity",
            {
                "repository_id": verified_assets.manifest.repository_id,
                "revision": verified_assets.manifest.revision,
                "root": str(verified_assets.root),
                "manifest": str(verified_assets.manifest_path),
                "config_name": verified_assets.manifest.runtime.config_name,
                "asset_id": verified_assets.manifest.runtime.asset_id,
                "norm_stats": str(verified_assets.norm_stats_path),
                "action_horizon": verified_assets.manifest.runtime.action_horizon,
                "raw_state_dim": verified_assets.manifest.runtime.state_dim,
                "model_state_dim": verified_assets.manifest.runtime.model_state_dim,
                "environment_action_dim": (
                    verified_assets.manifest.runtime.environment_action_dim
                ),
                "model_action_dim": (verified_assets.manifest.runtime.model_action_dim),
                "camera_keys": (
                    verified_assets.manifest.runtime.cameras.observation_keys
                ),
            },
        )
    return policy
