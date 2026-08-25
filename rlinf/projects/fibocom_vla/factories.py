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

import importlib
import operator
import os
from collections.abc import Callable
from typing import Any

from .config import StackConfig
from .contracts import ChunkPolicy
from .errors import ConfigurationError


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required")
    return value


def _load_factory(path: str, *, variable: str) -> Callable[..., Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ConfigurationError(f"{variable} must use module:function syntax")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except Exception as error:
        raise ConfigurationError(
            f"failed to import {variable}={path!r}: {error}"
        ) from error
    if not callable(factory):
        raise ConfigurationError(f"{variable} target is not callable")
    return factory


def _camera_mapping(config: StackConfig) -> tuple[str, str | None]:
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
    return main, wrist


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


def _validate_openpi_shape_contract(model: Any, config: StackConfig) -> None:
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
    if padded_action_dim < expected_action_dim:
        raise ConfigurationError(
            "OpenPI padded action_dim is smaller than the requested environment "
            f"action dimension: {padded_action_dim} < {expected_action_dim}"
        )


def create_openpi_policy_from_env(config: StackConfig):
    """Load an RLinf π0.5 checkpoint using explicit environment variables.

    Required:
        ``FIBOCOM_PI05_MODEL_PATH`` points to an OpenPI checkpoint containing
        the normalization statistics expected by RLinf.

    ``FIBOCOM_PI05_CONFIG_NAME`` is also required: checkpoint preprocessing
    semantics must never be guessed. ``FIBOCOM_PI05_DEVICE`` defaults to
    ``cuda``. Camera keys are checked against the action-adapter contract.

    When residual/speculative layers are enabled, their assets and factories
    are mandatory as well. Missing integration never silently degrades to a
    naked base policy.
    """

    model_path = _required_env("FIBOCOM_PI05_MODEL_PATH")
    device = os.environ.get("FIBOCOM_PI05_DEVICE", "cuda")
    config_name = _required_env("FIBOCOM_PI05_CONFIG_NAME")
    main_camera, wrist_camera = _camera_mapping(config)
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi import get_model

    from .openpi_adapter import OpenPiAdapterConfig, RLinfOpenPiChunkPolicy

    model_config = OmegaConf.create(
        {
            "model_path": model_path,
            "openpi": {
                "config_name": config_name,
                "action_chunk": config.residual_rl.action_horizon,
                "action_env_dim": config.residual_rl.action_dim,
                "num_steps": int(os.environ.get("FIBOCOM_PI05_DENOISE_STEPS", "5")),
                "train_expert_only": True,
                "add_value_head": False,
                "noise_method": os.environ.get("FIBOCOM_PI05_NOISE_METHOD", "flow_sde"),
            },
        }
    )
    model = get_model(model_config)
    _validate_openpi_shape_contract(model, config)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_policy = RLinfOpenPiChunkPolicy(
        model,
        OpenPiAdapterConfig(
            main_camera=main_camera,
            wrist_camera=wrist_camera,
            period_s=1.0 / config.robot.control_hz,
        ),
    )
    policy: ChunkPolicy = base_policy
    layers = ["rlinf_openpi_pi05"]

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
        draft_path = _required_env("FIBOCOM_DRAFT_POLICY_FACTORY")
        verifier_path = _required_env("FIBOCOM_PARALLEL_VERIFIER_FACTORY")
        draft_policy = _load_factory(
            draft_path, variable="FIBOCOM_DRAFT_POLICY_FACTORY"
        )(config)
        if not isinstance(draft_policy, ChunkPolicy):
            raise ConfigurationError(
                "FIBOCOM_DRAFT_POLICY_FACTORY did not return a ChunkPolicy"
            )
        verifier = _load_factory(
            verifier_path, variable="FIBOCOM_PARALLEL_VERIFIER_FACTORY"
        )(config, base_policy, policy)
        if not callable(getattr(verifier, "verify", None)):
            raise ConfigurationError(
                "FIBOCOM_PARALLEL_VERIFIER_FACTORY result lacks verify()"
            )
        from .inference.speculative import SpeculativeChunkPolicy

        policy = SpeculativeChunkPolicy(
            draft_policy,
            policy,
            verifier,
            config.speculative,
            resume_override=True,
        )
        layers.append("continuous_speculative_parallel_verify")

    setattr(policy, "fibocom_stack_layers", tuple(layers))
    setattr(policy, "fibocom_native_rtc_model_lineage", policy is base_policy)
    return policy
