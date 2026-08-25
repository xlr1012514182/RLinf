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
# openpi model configs

import os

import torch
from omegaconf import DictConfig

from rlinf.projects.fibocom_vla.assets import (
    INFERENCE_IGNORED_AUXILIARY_KIND,
    checkpoint_load_key_classes,
)

CHECKPOINT_LOAD_REPORT_ATTR = "_rlinf_checkpoint_load_report"

INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA = {
    "value_head.mlp.0.bias": ((512,), torch.float32),
    "value_head.mlp.0.weight": ((512, 1024), torch.float32),
    "value_head.mlp.2.bias": ((256,), torch.float32),
    "value_head.mlp.2.weight": ((256, 512), torch.float32),
    "value_head.mlp.4.bias": ((128,), torch.float32),
    "value_head.mlp.4.weight": ((128, 256), torch.float32),
    "value_head.mlp.6.bias": ((1,), torch.float32),
    "value_head.mlp.6.weight": ((1, 128), torch.float32),
}


def _inference_value_head_auxiliary_keys(model, state_dict, source_kind):
    """Classify only the exact, source-evidenced RoboTwin training value head."""

    config = getattr(model, "config", None)
    required_model_fields = {
        "config_name": "pi05_aloha_robotwin",
        "pi05": True,
        "action_horizon": 50,
        "action_dim": 32,
        "add_value_head": False,
        "value_after_vlm": False,
    }
    if source_kind != "safetensors_shards" or config is None:
        return ()
    if any(
        getattr(config, name, None) != expected
        for name, expected in required_model_fields.items()
    ):
        return ()
    observed = {key for key in state_dict if key.startswith("value_head.")}
    expected = set(INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA)
    if observed != expected:
        return ()
    for key, (shape, dtype) in INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA.items():
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            return ()
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            return ()
    return tuple(sorted(expected))


def _load_state_dict_with_report(model, state_dict, *, source_kind, selected_paths):
    """Load the exact main model and explicitly classify one training auxiliary."""

    ignored_auxiliary = _inference_value_head_auxiliary_keys(
        model, state_dict, source_kind
    )
    if ignored_auxiliary:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in set(ignored_auxiliary)
        }
    incompatible = model.load_state_dict(state_dict, strict=False)
    unresolved = tuple(incompatible.unexpected_keys)
    raw_unexpected = tuple(sorted((*unresolved, *ignored_auxiliary)))
    report = {
        "source_kind": str(source_kind),
        "selected_paths": tuple(os.path.abspath(path) for path in selected_paths),
        "missing_keys": tuple(incompatible.missing_keys),
        # Keep this raw: ignored training auxiliaries remain visible evidence.
        "unexpected_keys": raw_unexpected,
        "ignored_auxiliary_keys": ignored_auxiliary,
        "ignored_auxiliary_kind": (
            INFERENCE_IGNORED_AUXILIARY_KIND if ignored_auxiliary else None
        ),
        "unresolved_unexpected_keys": unresolved,
    }
    checkpoint_load_key_classes(report)
    setattr(model, CHECKPOINT_LOAD_REPORT_ATTR, report)
    return report


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    actor_train_config = get_openpi_config(config_name, model_path=cfg.model_path)
    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_config_kwargs = cfg.openpi
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            actor_model_config.__dict__[key] = val
    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        _load_state_dict_with_report(
            model,
            model_state_dict,
            source_kind="model_state_dict_full_weights",
            selected_paths=(full_weights_path,),
        )
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        _load_state_dict_with_report(
            model,
            model_state_dict,
            source_kind="actor_model_state_dict_full_weights",
            selected_paths=(actor_full_weights_path,),
        )
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            duplicate_keys = sorted(set(all_state_dict).intersection(state_dict))
            if duplicate_keys:
                raise ValueError(
                    "duplicate tensors across safetensors shards: "
                    + ", ".join(duplicate_keys[:10])
                )
            all_state_dict.update(state_dict)
        _load_state_dict_with_report(
            model,
            all_state_dict,
            source_kind="safetensors_shards",
            selected_paths=tuple(weight_paths),
        )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    load_report = dict(getattr(model, CHECKPOINT_LOAD_REPORT_ATTR))
    load_report.update(
        {
            "data_asset_id": data_config.asset_id,
            "use_quantile_norm": bool(data_config.use_quantile_norm),
        }
    )
    setattr(model, CHECKPOINT_LOAD_REPORT_ATTR, load_report)
    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
