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

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from rlinf.projects.fibocom_vla.config import (
    ResidualRLConfig,
    RTCConfig,
    SpeculativeConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError
from rlinf.projects.fibocom_vla.factories import (
    _validate_openpi_load_report,
    create_openpi_policy_from_env,
)
from rlinf.projects.fibocom_vla.openpi_adapter import RLinfOpenPiChunkPolicy

_OPENPI_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "rlinf"
    / "models"
    / "embodiment"
    / "openpi"
    / "__init__.py"
)
_OPENPI_SPEC = importlib.util.spec_from_file_location(
    "_fibocom_factory_openpi_loader", _OPENPI_SOURCE
)
assert _OPENPI_SPEC is not None and _OPENPI_SPEC.loader is not None
_OPENPI_LOADER = importlib.util.module_from_spec(_OPENPI_SPEC)
_OPENPI_SPEC.loader.exec_module(_OPENPI_LOADER)
INFERENCE_IGNORED_AUXILIARY_KIND = _OPENPI_LOADER.INFERENCE_IGNORED_AUXILIARY_KIND
INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA = (
    _OPENPI_LOADER.INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA
)
checkpoint_load_key_classes = _OPENPI_LOADER.checkpoint_load_key_classes


class _FakeModel:
    def __init__(
        self,
        *,
        action_horizon: int = 50,
        action_chunk: int = 50,
        action_env_dim: int = 7,
        action_dim: int = 32,
        config_name: str = "unit_test_pi05",
        pi05: bool = True,
        discrete_state_input: bool = True,
        max_token_len: int = 200,
        num_images_in_input: int = 1,
    ) -> None:
        self.training = True
        self.config = types.SimpleNamespace(
            config_name=config_name,
            pi05=pi05,
            discrete_state_input=discrete_state_input,
            action_horizon=action_horizon,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
            action_dim=action_dim,
            max_token_len=max_token_len,
            num_images_in_input=num_images_in_input,
        )
        self._rlinf_checkpoint_load_report = {
            "source_kind": "safetensors_shards",
            "selected_paths": (),
            "missing_keys": (),
            "unexpected_keys": (),
            "data_asset_id": "physical-intelligence/unit-test",
            "use_quantile_norm": True,
        }

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return ()


def _base_only_config() -> StackConfig:
    return StackConfig(
        residual_rl=ResidualRLConfig(enabled=False),
        speculative=SpeculativeConfig(enabled=False),
    )


def test_factory_accepts_only_classified_training_value_head_auxiliary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = (tmp_path / "weights.safetensors").resolve()
    auxiliary = tuple(sorted(INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA))
    report = {
        "source_kind": "safetensors_shards",
        "selected_paths": (str(weights),),
        "missing_keys": (),
        "unexpected_keys": auxiliary,
        "ignored_auxiliary_keys": auxiliary,
        "ignored_auxiliary_kind": INFERENCE_IGNORED_AUXILIARY_KIND,
        "unresolved_unexpected_keys": (),
        "data_asset_id": "physical-intelligence/robotwin",
        "use_quantile_norm": True,
    }
    model = types.SimpleNamespace(_rlinf_checkpoint_load_report=report)
    verified = types.SimpleNamespace(
        manifest=types.SimpleNamespace(
            runtime=types.SimpleNamespace(asset_id="physical-intelligence/robotwin")
        ),
        weight_paths=(weights,),
    )
    loader_module = types.ModuleType("rlinf.models.embodiment.openpi")
    loader_module.checkpoint_load_key_classes = checkpoint_load_key_classes
    monkeypatch.setitem(sys.modules, "rlinf.models.embodiment.openpi", loader_module)

    _validate_openpi_load_report(model, verified)

    report["unexpected_keys"] = (*auxiliary, "rogue.weight")
    report["unresolved_unexpected_keys"] = ("rogue.weight",)
    with pytest.raises(ConfigurationError, match="main state_dict"):
        _validate_openpi_load_report(model, verified)


def _patch_model_loader(
    monkeypatch,
    *,
    registered_overrides: dict[str, object] | None = None,
    missing_keys: tuple[str, ...] = (),
    unexpected_keys: tuple[str, ...] = (),
    use_quantile_norm: bool = True,
    data_asset_id: str = "physical-intelligence/unit-test",
    **model_kwargs,
) -> _FakeModel:
    model = _FakeModel(**model_kwargs)
    model._rlinf_checkpoint_load_report["missing_keys"] = missing_keys
    model._rlinf_checkpoint_load_report["unexpected_keys"] = unexpected_keys
    model._rlinf_checkpoint_load_report["use_quantile_norm"] = use_quantile_norm
    model._rlinf_checkpoint_load_report["data_asset_id"] = data_asset_id
    models = types.ModuleType("rlinf.models")
    models.__path__ = []
    embodiment = types.ModuleType("rlinf.models.embodiment")
    embodiment.__path__ = []
    openpi = types.ModuleType("rlinf.models.embodiment.openpi")
    openpi.__path__ = []
    openpi.checkpoint_load_key_classes = checkpoint_load_key_classes

    def get_model(config):
        model.received_config = config
        for name, value in config.openpi.items():
            setattr(model.config, name, value)
        model._rlinf_checkpoint_load_report["selected_paths"] = (
            str(Path(str(config.model_path)).resolve() / "weights.safetensors"),
        )
        return model

    openpi.get_model = get_model
    dataconfig = types.ModuleType("rlinf.models.embodiment.openpi.dataconfig")
    registered_values = {
        "name": "unit_test_pi05",
        "pi05": True,
        "discrete_state_input": True,
        "action_horizon": 50,
        "action_dim": 32,
        "max_token_len": 200,
        "repo_id": "physical-intelligence/unit-test",
        "asset_id": None,
        "adapt_to_pi": True,
        "extra_delta_transform": True,
    }
    registered_values.update(registered_overrides or {})

    def get_openpi_config(config_name, model_path):
        del config_name, model_path
        return types.SimpleNamespace(
            name=registered_values["name"],
            model=types.SimpleNamespace(
                pi05=registered_values["pi05"],
                discrete_state_input=registered_values["discrete_state_input"],
                action_horizon=registered_values["action_horizon"],
                action_dim=registered_values["action_dim"],
                max_token_len=registered_values["max_token_len"],
            ),
            data=types.SimpleNamespace(
                repo_id=registered_values["repo_id"],
                assets=types.SimpleNamespace(asset_id=registered_values["asset_id"]),
                adapt_to_pi=registered_values["adapt_to_pi"],
                extra_delta_transform=registered_values["extra_delta_transform"],
            ),
        )

    dataconfig.get_openpi_config = get_openpi_config
    monkeypatch.setitem(sys.modules, "rlinf.models", models)
    monkeypatch.setitem(sys.modules, "rlinf.models.embodiment", embodiment)
    monkeypatch.setitem(sys.modules, "rlinf.models.embodiment.openpi", openpi)
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi.dataconfig",
        dataconfig,
    )
    return model


def test_openpi_factory_requires_explicit_checkpoint_config(monkeypatch) -> None:
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", "unit-test-checkpoint")
    monkeypatch.delenv("FIBOCOM_PI05_CONFIG_NAME", raising=False)

    with pytest.raises(ConfigurationError, match="CONFIG_NAME"):
        create_openpi_policy_from_env(_base_only_config())


def test_openpi_factory_builds_only_explicitly_enabled_layers(monkeypatch) -> None:
    model = _patch_model_loader(monkeypatch)
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", "unit-test-checkpoint")
    monkeypatch.setenv("FIBOCOM_PI05_CONFIG_NAME", "unit_test_pi05")
    monkeypatch.setenv("FIBOCOM_PI05_DEVICE", "cpu")

    policy = create_openpi_policy_from_env(_base_only_config())

    assert isinstance(policy, RLinfOpenPiChunkPolicy)
    assert policy.model is model
    assert model.device == "cpu"
    assert policy.fibocom_stack_layers == ("rlinf_openpi_pi05",)
    assert policy.fibocom_native_rtc_model_lineage is True


def test_openpi_factory_opt_in_installs_concrete_visual_graph(monkeypatch) -> None:
    model = _patch_model_loader(monkeypatch)
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", "unit-test-checkpoint")
    monkeypatch.setenv("FIBOCOM_PI05_CONFIG_NAME", "unit_test_pi05")
    monkeypatch.setenv("FIBOCOM_PI05_DEVICE", "cuda:1")
    monkeypatch.setenv("FIBOCOM_PI05_CUDA_GRAPH_VISUAL", "true")
    monkeypatch.setenv("FIBOCOM_PI05_CUDA_GRAPH_CACHE_CAPACITY", "2")
    monkeypatch.setenv("FIBOCOM_PI05_CUDA_GRAPH_WARMUP", "5")
    import rlinf.projects.fibocom_vla.inference.openpi_cuda_graph as graph_module

    calls = []
    installed_patch = object()

    def install(model_arg, **kwargs):
        calls.append((model_arg, kwargs))
        return installed_patch

    monkeypatch.setattr(graph_module, "install_rlinf_openpi_visual_graph", install)
    config = StackConfig(
        residual_rl=ResidualRLConfig(enabled=False),
        speculative=SpeculativeConfig(enabled=False),
        rtc=RTCConfig(enabled=False),
    )

    policy = create_openpi_policy_from_env(config)

    assert calls and calls[0][0] is model
    assert calls[0][1]["cache_capacity"] == 2
    assert calls[0][1]["warmup_iterations"] == 5
    context = calls[0][1]["default_context"]
    assert context.rtc is False
    assert context.dynamic_branch is False
    assert policy.fibocom_visual_graph_patch is installed_patch
    assert policy.fibocom_stack_layers == (
        "rlinf_openpi_pi05",
        "cuda_graph_visual_static_eager_guard",
    )


def test_openpi_factory_rejects_visual_graph_on_cpu(monkeypatch) -> None:
    _patch_model_loader(monkeypatch)
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", "unit-test-checkpoint")
    monkeypatch.setenv("FIBOCOM_PI05_CONFIG_NAME", "unit_test_pi05")
    monkeypatch.setenv("FIBOCOM_PI05_DEVICE", "cpu")
    monkeypatch.setenv("FIBOCOM_PI05_CUDA_GRAPH_VISUAL", "yes")

    with pytest.raises(ConfigurationError, match="requires a CUDA model device"):
        create_openpi_policy_from_env(_base_only_config())


def test_openpi_factory_rejects_short_checkpoint_horizon(monkeypatch) -> None:
    _patch_model_loader(monkeypatch, action_horizon=10)
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", "unit-test-checkpoint")
    monkeypatch.setenv("FIBOCOM_PI05_CONFIG_NAME", "pi05_libero")
    monkeypatch.setenv("FIBOCOM_PI05_DEVICE", "cpu")

    with pytest.raises(
        ConfigurationError,
        match=r"action_horizon mismatch: loaded 10.*requires 50",
    ):
        create_openpi_policy_from_env(_base_only_config())


def test_enabled_layers_never_silently_degrade_when_assets_are_missing(
    monkeypatch,
) -> None:
    _patch_model_loader(monkeypatch)
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", "unit-test-checkpoint")
    monkeypatch.setenv("FIBOCOM_PI05_CONFIG_NAME", "unit_test_pi05")
    monkeypatch.setenv("FIBOCOM_PI05_DEVICE", "cpu")
    monkeypatch.delenv("FIBOCOM_RESIDUAL_CHECKPOINT", raising=False)
    config = StackConfig(speculative=SpeculativeConfig(enabled=False))

    with pytest.raises(ConfigurationError, match="RESIDUAL_CHECKPOINT"):
        create_openpi_policy_from_env(config)

    config = StackConfig(
        residual_rl=ResidualRLConfig(enabled=False),
        speculative=SpeculativeConfig(enabled=True),
    )
    monkeypatch.delenv("FIBOCOM_PI05_ASSET_MANIFEST", raising=False)
    with pytest.raises(ConfigurationError, match="PI05_ASSET_MANIFEST"):
        create_openpi_policy_from_env(config)


def test_production_speculative_rejects_residual_policy_composition() -> None:
    config = StackConfig(
        residual_rl=ResidualRLConfig(enabled=True),
        speculative=SpeculativeConfig(enabled=True),
    )

    with pytest.raises(
        ConfigurationError,
        match="residual_rl and speculative cannot be enabled together",
    ):
        create_openpi_policy_from_env(config)


def test_production_speculative_loads_only_signed_manifest_bound_draft(
    monkeypatch, tmp_path: Path
) -> None:
    model = _patch_model_loader(
        monkeypatch,
        action_horizon=50,
        action_chunk=50,
        action_env_dim=14,
        num_images_in_input=3,
    )
    semantics = tuple(f"joint_{index}" for index in range(14))
    runtime = types.SimpleNamespace(
        config_name="unit_test_pi05",
        asset_id="physical-intelligence/unit-test",
        action_horizon=50,
        model_action_dim=32,
        max_token_length=200,
        discrete_state_input=True,
        environment_action_dim=14,
        state_dim=14,
        model_state_dim=32,
        normalization="quantile_q01_q99",
        adapt_to_pi=True,
        extra_delta_transform=True,
        state_semantics=semantics,
        cameras=types.SimpleNamespace(
            main="cam_high",
            wrists=("cam_left_wrist", "cam_right_wrist"),
            observation_keys=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        ),
    )
    verified_assets = types.SimpleNamespace(
        root=tmp_path.resolve(),
        manifest=types.SimpleNamespace(
            runtime=runtime,
            repository_id="unit/test-pi05",
            revision="a" * 40,
        ),
        manifest_path=(tmp_path / "main-manifest.json").resolve(),
        norm_stats_path=(tmp_path / "norm-stats.json").resolve(),
        weight_paths=((tmp_path / "weights.safetensors").resolve(),),
    )
    import rlinf.projects.fibocom_vla.factories as factories_module
    import rlinf.projects.fibocom_vla.inference.openpi_production_speculative as production_module
    import rlinf.projects.fibocom_vla.inference.openpi_speculative as speculative_module
    import rlinf.projects.fibocom_vla.inference.pi05_draft as draft_module

    monkeypatch.setattr(
        factories_module,
        "_asset_bound_model_inputs",
        lambda config: (
            str(tmp_path.resolve()),
            "cam_high",
            ("cam_left_wrist", "cam_right_wrist"),
            verified_assets,
        ),
    )
    verified_target = object()
    verified_target_calls = []

    def issue_target(cls, loaded_model, assets):
        verified_target_calls.append((loaded_model, assets))
        return verified_target

    monkeypatch.setattr(
        speculative_module.VerifiedOpenPITarget,
        "from_verified_assets",
        classmethod(issue_target),
    )
    expected_draft_target = object()
    target_contract_calls = []

    def derive_target(cls, target, *, draft_parameter_dtype):
        target_contract_calls.append((target, draft_parameter_dtype))
        return expected_draft_target

    monkeypatch.setattr(
        draft_module.Pi05DraftTargetContract,
        "from_verified_openpi",
        classmethod(derive_target),
    )
    draft_adapter = object()
    loader_calls = []

    def load_draft(**kwargs):
        loader_calls.append(kwargs)
        return draft_adapter

    monkeypatch.setattr(draft_module, "load_production_pi05_draft", load_draft)
    construction_calls = []

    class _FakeProductionPolicy:
        def __init__(self, *args, **kwargs):
            construction_calls.append((args, kwargs))

    monkeypatch.setattr(
        production_module,
        "OpenPIProductionSpeculativePolicy",
        _FakeProductionPolicy,
    )

    draft_manifest = tmp_path / "draft-manifest.json"
    draft_approval = tmp_path / "draft-approval.json"
    draft_evaluation = tmp_path / "draft-evaluation.json"
    public_key = tmp_path / "draft-public-key.bin"
    for path in (draft_manifest, draft_approval, draft_evaluation):
        path.write_text("{}\n", encoding="utf-8")
    public_key.write_bytes(b"k" * 32)
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", str(tmp_path.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_ASSET_MANIFEST", "main-manifest.json")
    monkeypatch.setenv("FIBOCOM_PI05_DEVICE", "cpu")
    monkeypatch.setenv("FIBOCOM_PI05_DRAFT_MANIFEST", str(draft_manifest.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_DRAFT_APPROVAL", str(draft_approval.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_DRAFT_EVALUATION", str(draft_evaluation.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_DRAFT_PUBLIC_KEY", str(public_key.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_DRAFT_SIGNING_KEY_ID", "unit-ed25519-key")
    monkeypatch.setenv("FIBOCOM_PI05_DRAFT_DTYPE", "float32")
    monkeypatch.setenv("FIBOCOM_PI05_SPECULATIVE_VERIFICATION_TIMES", "0.2, 0.1")
    monkeypatch.setenv("FIBOCOM_PI05_SPECULATIVE_BACKEND", "torch")
    monkeypatch.setenv("FIBOCOM_PI05_SPECULATIVE_EPISODE_ID", "unit-episode")
    # Obsolete arbitrary-callable variables must have no effect on assembly.
    monkeypatch.setenv("FIBOCOM_DRAFT_POLICY_FACTORY", "evil.module:draft")
    monkeypatch.setenv("FIBOCOM_PARALLEL_VERIFIER_FACTORY", "evil.module:verify")
    config = StackConfig(
        residual_rl=ResidualRLConfig(
            enabled=False,
            action_horizon=50,
            action_dim=14,
        ),
        speculative=SpeculativeConfig(enabled=True),
    )

    policy = create_openpi_policy_from_env(config)

    assert verified_target_calls == [(model, verified_assets)]
    assert target_contract_calls == [(verified_target, "float32")]
    assert loader_calls == [
        {
            "candidate_manifest_path": str(draft_manifest.resolve()),
            "approval_path": str(draft_approval.resolve()),
            "evaluation_report_path": str(draft_evaluation.resolve()),
            "expected_target": expected_draft_target,
            "trusted_public_key_raw": b"k" * 32,
            "trusted_signing_key_id": "unit-ed25519-key",
            "device": "cpu",
        }
    ]
    assert len(construction_calls) == 1
    args, kwargs = construction_calls[0]
    assert args[:2] == (verified_target, draft_adapter)
    assert isinstance(args[2], RLinfOpenPiChunkPolicy)
    assert args[2].model is model
    assert args[3] is config.speculative
    assert kwargs == {
        "episode_id": "unit-episode",
        "verification_times": (0.2, 0.1),
        "backend": "torch",
    }
    assert policy.fibocom_stack_layers == (
        "rlinf_openpi_pi05",
        "openpi_production_signed_draft_parallel_verify",
    )
    assert policy.fibocom_native_rtc_model_lineage is False
    assert policy.fibocom_verified_openpi_target is verified_target


def test_manifest_bound_factory_uses_verified_geometry_and_identity(
    monkeypatch, tmp_path: Path
) -> None:
    model = _patch_model_loader(
        monkeypatch,
        action_horizon=50,
        action_chunk=50,
        action_env_dim=7,
    )
    runtime = types.SimpleNamespace(
        model_family="pi05",
        config_name="unit_test_pi05",
        asset_id="physical-intelligence/unit-test",
        action_horizon=50,
        model_action_dim=32,
        max_token_length=200,
        discrete_state_input=True,
        environment_action_dim=7,
        state_dim=7,
        model_state_dim=32,
        normalization="quantile_q01_q99",
        adapt_to_pi=True,
        extra_delta_transform=True,
        action_mode="delta_except_gripper",
        policy_frame="joint",
        state_semantics=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
            "aux",
        ),
        environment_action_semantics=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
            "aux",
        ),
        delta_action_mask=(True, True, True, True, True, False, True),
        norm_stats_path=Path("norm_stats.json"),
        source_fingerprints=(
            types.SimpleNamespace(
                path=Path("runtime.py"),
                sha256_lf="1" * 64,
                repository="https://example.invalid/rlinf",
                revision="b" * 40,
            ),
        ),
        cameras=types.SimpleNamespace(
            main="main",
            wrists=(),
            observation_keys=("main",),
            model_image_keys=("base_0_rgb",),
        ),
    )
    manifest = types.SimpleNamespace(
        runtime=runtime,
        repository_id="unit/test-pi05",
        revision="a" * 40,
        validate_stack=lambda config: None,
        files=(
            types.SimpleNamespace(
                role="selected_norm_stats",
                path=Path("norm_stats.json"),
                sha256="2" * 64,
            ),
            types.SimpleNamespace(
                role="weights_shard",
                path=Path("weights.safetensors"),
                sha256="3" * 64,
            ),
        ),
    )
    verified = types.SimpleNamespace(
        root=tmp_path.resolve(),
        manifest=manifest,
        manifest_path=(tmp_path / "manifest.json").resolve(),
        norm_stats_path=(tmp_path / "norm_stats.json").resolve(),
        weight_paths=((tmp_path / "weights.safetensors").resolve(),),
        checkpoint_id=f"unit/test-pi05@{'a' * 40}",
    )
    import rlinf.projects.fibocom_vla.assets as assets

    calls = []

    def load_assets(manifest_path, root, *, repository_root):
        calls.append((manifest_path, Path(root), Path(repository_root)))
        return verified

    monkeypatch.setattr(assets, "load_and_verify_checkpoint_assets", load_assets)
    monkeypatch.setattr(
        assets,
        "require_verified_checkpoint_assets",
        lambda value: value,
    )
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", str(tmp_path.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_ASSET_MANIFEST", "unit-manifest.json")
    monkeypatch.delenv("FIBOCOM_PI05_CONFIG_NAME", raising=False)
    monkeypatch.setenv("FIBOCOM_PI05_DEVICE", "cpu")

    policy = create_openpi_policy_from_env(_base_only_config())

    assert calls and calls[0][0] == "unit-manifest.json"
    assert model.received_config.openpi.action_horizon == 50
    assert model.received_config.openpi.num_images_in_input == 1
    assert policy.config.main_camera == "main"
    assert policy.config.resolved_wrist_cameras == ()
    assert policy.config.expected_state_names == runtime.state_semantics
    assert policy.fibocom_checkpoint_identity == {
        "repository_id": "unit/test-pi05",
        "revision": "a" * 40,
        "root": str(tmp_path.resolve()),
        "manifest": str((tmp_path / "manifest.json").resolve()),
        "config_name": "unit_test_pi05",
        "asset_id": "physical-intelligence/unit-test",
        "norm_stats": str((tmp_path / "norm_stats.json").resolve()),
        "action_horizon": 50,
        "raw_state_dim": 7,
        "model_state_dim": 32,
        "environment_action_dim": 7,
        "model_action_dim": 32,
        "camera_keys": ("main",),
    }


def test_manifest_bound_factory_rejects_semantic_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", str(tmp_path.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_ASSET_MANIFEST", "unit-manifest.json")
    monkeypatch.setenv("FIBOCOM_PI05_CONFIG_NAME", "pi05_libero")
    runtime = types.SimpleNamespace(
        config_name="pi05_aloha_robotwin",
        cameras=types.SimpleNamespace(
            main="cam_high",
            wrists=("cam_left_wrist", "cam_right_wrist"),
            observation_keys=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        ),
    )
    manifest = types.SimpleNamespace(
        runtime=runtime, validate_stack=lambda config: None
    )
    verified = types.SimpleNamespace(root=tmp_path.resolve(), manifest=manifest)
    import rlinf.projects.fibocom_vla.assets as assets

    monkeypatch.setattr(
        assets,
        "load_and_verify_checkpoint_assets",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        assets,
        "require_verified_checkpoint_assets",
        lambda value: value,
    )

    with pytest.raises(ConfigurationError, match="conflicts"):
        create_openpi_policy_from_env(_base_only_config())


@pytest.mark.parametrize(
    ("registered_overrides", "match"),
    (
        ({"pi05": False}, "model.pi05"),
        ({"discrete_state_input": False}, "discrete_state_input"),
        ({"action_dim": 31}, "action_dim"),
        ({"max_token_len": 199}, "max_token_len"),
        ({"repo_id": "wrong/repository"}, "repo_id"),
        ({"adapt_to_pi": False}, "adapt_to_pi"),
        ({"extra_delta_transform": False}, "extra_delta_transform"),
    ),
)
def test_manifest_bound_factory_rejects_registered_contract_mismatch(
    monkeypatch,
    tmp_path: Path,
    registered_overrides: dict[str, object],
    match: str,
) -> None:
    _patch_model_loader(
        monkeypatch,
        registered_overrides=registered_overrides,
    )
    runtime = types.SimpleNamespace(
        config_name="unit_test_pi05",
        asset_id="physical-intelligence/unit-test",
        action_horizon=50,
        model_action_dim=32,
        max_token_length=200,
        discrete_state_input=True,
        environment_action_dim=7,
        normalization="quantile_q01_q99",
        adapt_to_pi=True,
        extra_delta_transform=True,
        state_semantics=StackConfig().robot.joint_names,
        cameras=types.SimpleNamespace(
            main="main", wrists=(), observation_keys=("main",)
        ),
    )
    verified = types.SimpleNamespace(
        root=tmp_path.resolve(),
        manifest=types.SimpleNamespace(
            runtime=runtime,
            validate_stack=lambda config: None,
        ),
    )
    import rlinf.projects.fibocom_vla.assets as assets

    monkeypatch.setattr(
        assets,
        "load_and_verify_checkpoint_assets",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        assets, "require_verified_checkpoint_assets", lambda value: value
    )
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", str(tmp_path.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_ASSET_MANIFEST", "unit-manifest.json")
    monkeypatch.delenv("FIBOCOM_PI05_CONFIG_NAME", raising=False)

    with pytest.raises(ConfigurationError, match=match):
        create_openpi_policy_from_env(_base_only_config())


def test_manifest_bound_factory_rejects_non_exact_checkpoint_keys(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_model_loader(monkeypatch, missing_keys=("model.missing_weight",))
    runtime = types.SimpleNamespace(
        config_name="unit_test_pi05",
        asset_id="physical-intelligence/unit-test",
        action_horizon=50,
        model_action_dim=32,
        max_token_length=200,
        discrete_state_input=True,
        environment_action_dim=7,
        normalization="quantile_q01_q99",
        adapt_to_pi=True,
        extra_delta_transform=True,
        state_semantics=StackConfig().robot.joint_names,
        cameras=types.SimpleNamespace(
            main="main", wrists=(), observation_keys=("main",)
        ),
    )
    manifest = types.SimpleNamespace(
        runtime=runtime,
        validate_stack=lambda config: None,
    )
    verified = types.SimpleNamespace(
        root=tmp_path.resolve(),
        manifest=manifest,
        weight_paths=((tmp_path / "weights.safetensors").resolve(),),
    )
    import rlinf.projects.fibocom_vla.assets as assets

    monkeypatch.setattr(
        assets,
        "load_and_verify_checkpoint_assets",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        assets, "require_verified_checkpoint_assets", lambda value: value
    )
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", str(tmp_path.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_ASSET_MANIFEST", "unit-manifest.json")
    monkeypatch.delenv("FIBOCOM_PI05_CONFIG_NAME", raising=False)

    with pytest.raises(ConfigurationError, match="not exact-key compatible"):
        create_openpi_policy_from_env(_base_only_config())


def test_manifest_bound_factory_rejects_loaded_non_quantile_preprocessing(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_model_loader(monkeypatch, use_quantile_norm=False)
    runtime = types.SimpleNamespace(
        config_name="unit_test_pi05",
        asset_id="physical-intelligence/unit-test",
        action_horizon=50,
        model_action_dim=32,
        max_token_length=200,
        discrete_state_input=True,
        environment_action_dim=7,
        normalization="quantile_q01_q99",
        adapt_to_pi=True,
        extra_delta_transform=True,
        state_semantics=StackConfig().robot.joint_names,
        cameras=types.SimpleNamespace(
            main="main", wrists=(), observation_keys=("main",)
        ),
    )
    verified = types.SimpleNamespace(
        root=tmp_path.resolve(),
        manifest=types.SimpleNamespace(
            runtime=runtime,
            validate_stack=lambda config: None,
        ),
        weight_paths=((tmp_path / "weights.safetensors").resolve(),),
    )
    import rlinf.projects.fibocom_vla.assets as assets

    monkeypatch.setattr(
        assets,
        "load_and_verify_checkpoint_assets",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        assets, "require_verified_checkpoint_assets", lambda value: value
    )
    monkeypatch.setenv("FIBOCOM_PI05_MODEL_PATH", str(tmp_path.resolve()))
    monkeypatch.setenv("FIBOCOM_PI05_ASSET_MANIFEST", "unit-manifest.json")
    monkeypatch.delenv("FIBOCOM_PI05_CONFIG_NAME", raising=False)

    with pytest.raises(ConfigurationError, match="use_quantile_norm"):
        create_openpi_policy_from_env(_base_only_config())
