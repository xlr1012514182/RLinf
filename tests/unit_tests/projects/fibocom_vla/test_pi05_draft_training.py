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
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from rlinf.projects.fibocom_vla import config as config_module
from rlinf.projects.fibocom_vla import factories
from rlinf.projects.fibocom_vla.contracts import Observation, RobotState
from rlinf.projects.fibocom_vla.errors import ConfigurationError
from rlinf.projects.fibocom_vla.inference import pi05_draft as pi05
from rlinf.projects.fibocom_vla.inference import pi05_draft_training as training
from rlinf.projects.fibocom_vla.inference.pi05_draft import (
    Pi05DraftTargetContract,
    WarmStartLineage,
)
from rlinf.projects.fibocom_vla.inference.pi05_draft_training import (
    DraftSourceSample,
    DraftTrainingConfig,
    TeacherMaterialization,
    export_candidate_from_run,
    materialize_teacher_cache,
    train_draft_model,
    verify_teacher_cache,
    verify_training_run,
)
from rlinf.projects.fibocom_vla.openpi_adapter import (
    OpenPiAdapterConfig,
    RLinfOpenPiChunkPolicy,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target() -> Pi05DraftTargetContract:
    return Pi05DraftTargetContract(
        parent_repository_id=pi05.PI05_PARENT_REPOSITORY_ID,
        parent_revision=pi05.PI05_PARENT_REVISION,
        parent_asset_manifest_sha256="1" * 64,
        norm_stats_sha256="2" * 64,
        transform_id="unit-test-transform",
        transform_source_sha256=("3" * 64,),
        draft_parameter_dtype="float32",
    )


def _observation(value: float, frame: int) -> Observation:
    names = tuple(f"joint_{index}" for index in range(14))
    return Observation(
        state=RobotState(
            joint_positions=np.full((14,), value, dtype=np.float32),
            joint_names=names,
        ),
        images={"cam_high": np.zeros((4, 4, 3), dtype=np.uint8)},
        instruction="stack the block",
        frame_id=frame,
        metadata={"unit_value": value},
    )


class _RawSource:
    def __init__(self, count: int = 4) -> None:
        self._samples = [
            DraftSourceSample(
                observation=_observation(float(index + 1), 0),
                episode_id=f"episode-{index}",
                frame_index=0,
                valid_steps=50 - index,
            )
            for index in range(count)
        ]

    def iter_samples(self) -> list[DraftSourceSample]:
        return list(self._samples)


class _FakeTeacherOwner:
    def __init__(self, target: Pi05DraftTargetContract) -> None:
        self.target_contract = target
        self.closed = False

    def materialize(
        self, observation: Observation, *, episode_id: str
    ) -> TeacherMaterialization:
        del episode_id
        value = float(observation.metadata["unit_value"])
        return TeacherMaterialization(
            prefix_embs=torch.full((2, 2048), value / 100, dtype=torch.float32),
            prefix_pad_masks=torch.ones(2, dtype=torch.bool),
            prefix_att_masks=torch.zeros(2, dtype=torch.bool),
            robot_state=torch.full((32,), value / 10, dtype=torch.float32),
            teacher_actions=torch.full((50, 32), value / 5, dtype=torch.float32),
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> training.VerifiedDraftCache:
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text('{"dataset":"unit-raw"}\n', encoding="utf-8")
    monkeypatch.setattr(training, "OpenPITeacherOwner", _FakeTeacherOwner)
    owner = _FakeTeacherOwner(_target())
    result = materialize_teacher_cache(
        _RawSource(),
        source_manifest_path=source_manifest,
        teacher_owner=owner,
        output_dir=tmp_path / "cache",
        validation_fraction=0.25,
    )
    assert owner.closed
    return result


class _TinyDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32, 50 * 32)

    def forward(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        robot_state: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> torch.Tensor:
        del prefix_embs, prefix_pad_masks, prefix_att_masks, last_actions
        self.last_input_dtype = robot_state.dtype
        return self.linear(robot_state).reshape(-1, 50, 32)


class _NaNDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        batch = int(inputs["robot_state"].shape[0])
        return self.weight * torch.full((batch, 50, 32), float("nan"))


def _tiny(seed: int = 123) -> _TinyDraft:
    torch.manual_seed(seed)
    return _TinyDraft()


def _config(steps: int = 2) -> DraftTrainingConfig:
    return DraftTrainingConfig(
        total_optimizer_steps=steps,
        training_code_repository="https://github.com/xlr1012514182/RLinf",
        training_code_revision="4" * 40,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        seed=7,
        device="cpu",
    )


def test_cache_binds_hash_shapes_and_episode_wise_split(
    cache: training.VerifiedDraftCache,
) -> None:
    checked = verify_teacher_cache(cache.manifest_path)
    manifest = checked.manifest

    assert checked.sample_count == 4
    train = set(manifest["split"]["train_episode_ids"])
    validation = set(manifest["split"]["validation_episode_ids"])
    assert train
    assert validation
    assert not train & validation
    for sample in manifest["samples"]:
        assert sample["sample_count"] == 1
        tensors = load_file(
            str(cache.manifest_path.parent / sample["path"]), device="cpu"
        )
        assert tensors["teacher_actions"].shape == (50, 32)
        assert tensors["robot_state"].shape == (32,)
        assert tensors["prefix_embs"].shape == (2, 2048)


def test_cache_rejects_hash_drift_and_path_traversal(
    cache: training.VerifiedDraftCache,
) -> None:
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    shard = cache.manifest_path.parent / manifest["samples"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"drift")
    with pytest.raises(ConfigurationError, match="size drifted"):
        verify_teacher_cache(cache.manifest_path)

    manifest["samples"][0]["path"] = "../escape.safetensors"
    cache.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="inside|escapes"):
        verify_teacher_cache(cache.manifest_path)


def test_source_cannot_inject_model_space_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadSource:
        def iter_samples(self) -> list[dict[str, Any]]:
            return [{"teacher_actions": torch.zeros(50, 32)}]

    source_manifest = tmp_path / "source.json"
    source_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(training, "OpenPITeacherOwner", _FakeTeacherOwner)
    with pytest.raises(ConfigurationError, match="cache injection is forbidden"):
        materialize_teacher_cache(
            BadSource(),
            source_manifest_path=source_manifest,
            teacher_owner=_FakeTeacherOwner(_target()),
            output_dir=tmp_path / "cache",
        )


def test_real_optimizer_step_changes_weight_and_is_internally_counted(
    cache: training.VerifiedDraftCache, tmp_path: Path
) -> None:
    model = _tiny()
    before = model.linear.weight.detach().clone()
    run = train_draft_model(
        cache,
        model,
        _config(steps=2),
        output_dir=tmp_path / "run",
        allow_test_model=True,
    )

    assert run.optimizer_steps == 2
    assert run.complete
    assert not torch.equal(before, model.linear.weight)
    assert run.manifest["production_authorized"] is False


def test_float32_cache_is_cast_to_model_parameter_dtype(
    cache: training.VerifiedDraftCache, tmp_path: Path
) -> None:
    model = _tiny().to(dtype=torch.float64)
    train_draft_model(
        cache,
        model,
        _config(steps=1),
        output_dir=tmp_path / "double-run",
        allow_test_model=True,
    )

    assert model.last_input_dtype == torch.float64


def test_nonfinite_forward_does_not_count_or_change_parameter(
    cache: training.VerifiedDraftCache, tmp_path: Path
) -> None:
    model = _NaNDraft()
    before = model.weight.detach().clone()
    with pytest.raises(ConfigurationError, match="NaN|infinity|non-finite"):
        train_draft_model(
            cache,
            model,
            _config(steps=1),
            output_dir=tmp_path / "nan-run",
            allow_test_model=True,
        )

    assert torch.equal(before, model.weight)
    assert not (tmp_path / "nan-run").exists()


def test_resume_matches_uninterrupted_optimizer_lineage(
    cache: training.VerifiedDraftCache, tmp_path: Path
) -> None:
    config = _config(steps=4)
    full_model = _tiny()
    train_draft_model(
        cache,
        full_model,
        config,
        output_dir=tmp_path / "full",
        allow_test_model=True,
    )

    partial_model = _tiny()
    partial = train_draft_model(
        cache,
        partial_model,
        config,
        output_dir=tmp_path / "partial",
        stop_after_new_steps=2,
        allow_test_model=True,
    )
    assert not partial.complete
    resumed_model = _tiny()
    resumed = train_draft_model(
        cache,
        resumed_model,
        config,
        output_dir=tmp_path / "resumed",
        resume_run=partial,
        allow_test_model=True,
    )

    assert resumed.complete
    assert resumed.manifest["parent_run_manifest_sha256"] == partial.manifest_sha256
    for full, resumed_value in zip(
        full_model.parameters(), resumed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(full, resumed_value, rtol=0, atol=0)


def test_resume_rejects_cache_or_config_hash_drift(
    cache: training.VerifiedDraftCache, tmp_path: Path
) -> None:
    config = _config(steps=3)
    partial = train_draft_model(
        cache,
        _tiny(),
        config,
        output_dir=tmp_path / "partial",
        stop_after_new_steps=1,
        allow_test_model=True,
    )
    drifted = DraftTrainingConfig(
        **{
            **config.to_dict(),
            "learning_rate": 2e-2,
            "loss": config.loss,
        }
    )
    with pytest.raises(ConfigurationError, match="config drifted"):
        train_draft_model(
            cache,
            _tiny(),
            drifted,
            output_dir=tmp_path / "resumed",
            resume_run=partial,
            allow_test_model=True,
        )


def test_caller_cannot_inject_steps_or_metrics(
    cache: training.VerifiedDraftCache, tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        train_draft_model(  # type: ignore[call-arg]
            cache,
            _tiny(),
            _config(),
            output_dir=tmp_path / "bad",
            allow_test_model=True,
            metrics={"passed": 1.0},
        )

    run = train_draft_model(
        cache,
        _tiny(),
        _config(steps=1),
        output_dir=tmp_path / "run",
        allow_test_model=True,
    )
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"]["validation_rms"] += 10
    run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="trainer-owned state"):
        verify_training_run(run.manifest_path)


def test_builtin_teacher_owner_rejects_nonmanifest_and_wrapped_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack_path = tmp_path / "stack.json"
    stack_path.write_text("{}", encoding="utf-8")
    options = {"stack_config": str(stack_path)}
    monkeypatch.delenv("FIBOCOM_PI05_ASSET_MANIFEST", raising=False)
    with pytest.raises(ConfigurationError, match="ASSET_MANIFEST"):
        training.create_teacher_owner_from_env(options)

    config = SimpleNamespace(
        residual_rl=SimpleNamespace(enabled=False),
        speculative=SimpleNamespace(enabled=False),
    )
    monkeypatch.setenv("FIBOCOM_PI05_ASSET_MANIFEST", str(tmp_path / "assets.json"))
    monkeypatch.setattr(
        config_module.StackConfig,
        "from_json",
        classmethod(lambda _cls, _path: config),
    )
    monkeypatch.setattr(
        factories,
        "create_openpi_policy_from_env",
        lambda _config: SimpleNamespace(wrapped=True),
    )
    with pytest.raises(ConfigurationError, match="unwrapped"):
        training.create_teacher_owner_from_env(options)

    bare = RLinfOpenPiChunkPolicy(object(), OpenPiAdapterConfig())
    monkeypatch.setattr(
        factories, "create_openpi_policy_from_env", lambda _config: bare
    )
    with pytest.raises(ConfigurationError, match="VerifiedOpenPITarget"):
        training.create_teacher_owner_from_env(options)


def test_dummy_scalar_step_cannot_replace_verified_run_for_export(
    cache: training.VerifiedDraftCache,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _TinyExportHead(cache.target_contract)
    head.record_completed_optimizer_step(torch.ones((), requires_grad=True).square())
    monkeypatch.setattr(training, "Pi05DraftChunkHead", _TinyExportHead)

    with pytest.raises(TypeError, match="VerifiedDraftTrainingRun"):
        export_candidate_from_run(
            None,  # type: ignore[arg-type]
            cache,
            head,
            output_dir=tmp_path / "candidate",
        )


class _TinyExportHead(_TinyDraft):
    def __init__(self, target: Pi05DraftTargetContract) -> None:
        super().__init__()
        self.target_contract = target
        self.warm_start_lineage = WarmStartLineage(
            checkpoint_filename="draft_libero_goal.pt",
            checkpoint_size=1,
            checkpoint_sha256=(
                "371d809745d688e03b2a09ccb89b509906530c7db48117236caf13e676333f17"
            ),
            initialization_seed=17,
        )
        self.production_compatible = False
        self._optimizer_steps = 0

    @property
    def training_step_count(self) -> int:
        return self._optimizer_steps

    def record_completed_optimizer_step(self, loss: torch.Tensor) -> None:
        assert loss.requires_grad and loss.grad_fn is not None
        self._optimizer_steps += 1


def test_export_recomputes_validation_and_stub_never_creates_approval(
    cache: training.VerifiedDraftCache,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "Pi05DraftChunkHead", _TinyExportHead)
    monkeypatch.setattr(training, "current_git_revision", lambda: ("4" * 40, False))
    head = _TinyExportHead(cache.target_contract)
    run = train_draft_model(
        cache,
        head,
        _config(steps=1),
        output_dir=tmp_path / "run",
    )
    captured: dict[str, Any] = {}

    def fake_low_level_export(
        exported_head: nn.Module, *, training_record: Any, output_dir: Path
    ) -> Any:
        captured["head"] = exported_head
        captured["record"] = training_record
        root = Path(output_dir)
        root.mkdir()
        manifest_path = root / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps({"production_authorized": False}), encoding="utf-8"
        )
        return SimpleNamespace(
            manifest_path=manifest_path,
            manifest_sha256=_sha256(manifest_path),
            weights_sha256="6" * 64,
        )

    monkeypatch.setattr(training, "export_pi05_draft_candidate", fake_low_level_export)
    export_head = _TinyExportHead(cache.target_contract)
    artifact = export_candidate_from_run(
        run,
        cache,
        export_head,
        output_dir=tmp_path / "candidate",
    )

    record = captured["record"]
    assert record.optimizer_steps == 1
    assert "held_out_rms" in record.metrics
    candidate = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert candidate == {"production_authorized": False}
    assert not list((tmp_path / "candidate").glob("*approval*"))


def test_export_rejects_manifest_and_state_metric_injection_after_recompute(
    cache: training.VerifiedDraftCache,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "Pi05DraftChunkHead", _TinyExportHead)
    monkeypatch.setattr(training, "current_git_revision", lambda: ("4" * 40, False))
    run = train_draft_model(
        cache,
        _TinyExportHead(cache.target_contract),
        _config(steps=1),
        output_dir=tmp_path / "run",
    )
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"]["validation_rms"] += 1.0
    state_path = run.manifest_path.parent / manifest["training_state"]["path"]
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["metrics"] = dict(manifest["metrics"])
    torch.save(state, state_path)
    manifest["training_state"]["size"] = state_path.stat().st_size
    manifest["training_state"]["sha256"] = _sha256(state_path)
    run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    forged = verify_training_run(run.manifest_path)

    with pytest.raises(ConfigurationError, match="does not match recomputation"):
        export_candidate_from_run(
            forged,
            cache,
            _TinyExportHead(cache.target_contract),
            output_dir=tmp_path / "candidate",
        )


def test_test_injected_run_cannot_export(
    cache: training.VerifiedDraftCache, tmp_path: Path
) -> None:
    run = train_draft_model(
        cache,
        _tiny(),
        _config(steps=1),
        output_dir=tmp_path / "run",
        allow_test_model=True,
    )
    with pytest.raises(TypeError, match="exact Pi05DraftChunkHead"):
        export_candidate_from_run(
            run,
            cache,
            _tiny(),  # type: ignore[arg-type]
            output_dir=tmp_path / "candidate",
        )
