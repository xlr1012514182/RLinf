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
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from rlinf.projects.fibocom_vla.draft_assets import (
    DraftAsset,
    DraftCompatibility,
    DraftResolution,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError, ShapeMismatchError
from rlinf.projects.fibocom_vla.inference import draft_head
from rlinf.projects.fibocom_vla.inference import pi05_draft as pi05
from rlinf.projects.fibocom_vla.inference.draft_head import (
    DraftChunkHead,
    load_draft_chunk_head,
)
from rlinf.projects.fibocom_vla.inference.pi05_draft import (
    PI05_DRAFT_PARAMETER_COUNT,
    Pi05DraftLossConfig,
    Pi05DraftTargetContract,
    Pi05DraftTrainingRecord,
    build_pi05_draft_loss_mask,
    derive_pi05_draft_from_warm_start,
    export_pi05_draft_candidate,
    load_production_pi05_draft,
    pi05_draft_distillation_loss,
    sign_pi05_draft_candidate,
)


def _target(**overrides: Any) -> Pi05DraftTargetContract:
    values: dict[str, Any] = {
        "parent_repository_id": pi05.PI05_PARENT_REPOSITORY_ID,
        "parent_revision": pi05.PI05_PARENT_REVISION,
        "parent_asset_manifest_sha256": "1" * 64,
        "norm_stats_sha256": (
            "bff5fcadaba626c28c1c2350d3baa7f1a04befc9f8b73f3e558a09c1b9c6ee52"
        ),
        "transform_id": "LeRobotAlohaDataConfig@12881eda+AlohaPolicy",
        "transform_source_sha256": (
            "2ceafb0798de7a9198ad8e12728f2af441f2c3f72c221b790b4f4867451797b1",
            "a1165b3d987fd021c67ce8722a2f86a150e3ebaf1340750ec6819bbe619c16c3",
        ),
        "draft_parameter_dtype": "float32",
    }
    values.update(overrides)
    return Pi05DraftTargetContract(**values)


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
    }


def _strict_warm_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DraftChunkHead:
    path = tmp_path / "draft_libero_goal_test.pt"
    state_dict = {
        name: torch.empty(
            shape,
            device="meta",
            dtype=draft_head._EXPECTED_STATE_DTYPES[name],
        )
        for name, shape in draft_head._EXPECTED_STATE_SHAPES.items()
    }
    torch.save({"draft_head": state_dict, "meta": _published_meta()}, path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    asset = DraftAsset(
        suite="libero_goal",
        filename=Path(path.name),
        size=len(payload),
        sha256=digest,
    )
    resolution = DraftResolution(
        verified_asset=asset.verify(tmp_path),
        compatibility=DraftCompatibility(
            purpose="warm_start_initialization_only",
            production_compatible=False,
            initialization_allowed=True,
            mismatches=("model_family: pi0 -> pi0.5",),
        ),
    )
    warm_start = load_draft_chunk_head(
        resolution,
        purpose="warm_start",
        device="meta",
        dtype=torch.float32,
    )
    monkeypatch.setattr(pi05, "_PUBLISHED_PARENT_SHA256S", frozenset({digest}))
    return warm_start


def _derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pi05.Pi05DraftChunkHead, Pi05DraftTargetContract]:
    target = _target()
    warm_start = _strict_warm_start(tmp_path, monkeypatch)
    head, report = derive_pi05_draft_from_warm_start(
        warm_start,
        target_contract=target,
        initialization_seed=17,
        device="meta",
    )
    assert report.warm_start_sha256 in pi05._PUBLISHED_PARENT_SHA256S
    return head, target


def _training_record(steps: int = 1) -> Pi05DraftTrainingRecord:
    return Pi05DraftTrainingRecord(
        dataset_manifest_sha256="2" * 64,
        training_code_repository="https://github.com/xlr1012514182/RLinf",
        training_code_revision="3" * 40,
        hparams={"learning_rate": 1e-4, "target_source": "teacher_zero_noise"},
        seed=7,
        optimizer_steps=steps,
        metrics={"held_out_weighted_huber": 0.25},
    )


def _record_one_step(head: pi05.Pi05DraftChunkHead) -> None:
    differentiable_loss = (torch.ones((), requires_grad=True) * 2).square()
    head.record_completed_optimizer_step(differentiable_loss)


def _fake_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    pi05.Pi05DraftChunkHead,
    Pi05DraftTargetContract,
    pi05.Pi05DraftCandidateArtifact,
    dict[str, str],
]:
    head, target = _derived(tmp_path, monkeypatch)
    _record_one_step(head)
    content_sha = "4" * 64
    captured_metadata: dict[str, str] = {}

    def fake_content_hash(_state_dict: Any) -> str:
        return content_sha

    def fake_save(path: Path, _state_dict: Any, metadata: dict[str, str]) -> None:
        captured_metadata.update(metadata)
        path.write_bytes(b"unit-test-safetensors")

    monkeypatch.setattr(pi05, "_state_dict_content_sha256", fake_content_hash)
    monkeypatch.setattr(pi05, "_save_safetensors", fake_save)
    artifact = export_pi05_draft_candidate(
        head,
        training_record=_training_record(),
        output_dir=tmp_path / "candidate",
    )
    return head, target, artifact, captured_metadata


def _write_evaluation_report(
    artifact: pi05.Pi05DraftCandidateArtifact,
    path: Path,
) -> Path:
    report = {
        "schema_version": 1,
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "target_contract_sha256": artifact.target_contract_sha256,
        "protocol_id": "unit-test-fixed-protocol",
        "passed": True,
        "metrics": {"unit_acceptance": 0.5},
    }
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def test_target_contract_pins_complete_robotwin_geometry_and_semantics() -> None:
    target = _target()

    assert target.action_horizon == 50
    assert target.model_action_dim == 32
    assert target.environment_action_dim == 14
    assert target.raw_state_dim == 14
    assert target.model_state_dim == 32
    assert target.prefix_embedding_dim == 2048
    assert target.max_token_length == 200
    assert target.raw_camera_keys == (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    assert target.environment_action_indices == tuple(range(14))
    assert target.gripper_indices == (6, 13)
    assert target.delta_action_mask[6] is False
    assert target.delta_action_mask[13] is False
    assert Pi05DraftTargetContract.from_dict(target.to_dict()) == target


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_repository_id", "someone/other-model"),
        ("config_name", "pi0_libero"),
        ("action_horizon", 10),
        ("model_action_dim", 14),
        ("raw_camera_keys", ("cam_high",)),
        ("gripper_indices", (6,)),
    ],
)
def test_target_contract_rejects_semantic_drift(field: str, value: Any) -> None:
    with pytest.raises(ConfigurationError, match="target contract mismatch"):
        _target(**{field: value})


def test_warm_start_copies_only_non_head_tensors_and_builds_exact_32d_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head, _target_contract = _derived(tmp_path, monkeypatch)

    assert sum(parameter.numel() for parameter in head.parameters()) == (
        PI05_DRAFT_PARAMETER_COUNT
    )
    assert head._action_head.weight.shape == (32, 2048)  # noqa: SLF001
    assert head._action_head.bias.shape == (32,)  # noqa: SLF001
    assert head.warm_start_lineage.reinitialized_keys == (
        "_action_head.bias",
        "_action_head.weight",
    )
    assert not any(
        name.startswith("_action_head") for name in head.warm_start_lineage.copied_keys
    )
    assert head.training_step_count == 0
    assert head.initialization_only is True
    with pytest.raises(ConfigurationError, match="untrained, unsigned"):
        head.production_predict_adapter()


def test_new_or_unpinned_7d_head_cannot_enter_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with torch.device("meta"):
        loose = DraftChunkHead(
            img_dim=2048,
            chunk_m=50,
            hidden_dim=16384,
            out_dim=7,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    with pytest.raises(ConfigurationError, match="strictly loaded checkpoint"):
        derive_pi05_draft_from_warm_start(
            loose,
            target_contract=_target(),
            initialization_seed=0,
            device="meta",
        )

    warm_start = _strict_warm_start(tmp_path, monkeypatch)
    monkeypatch.setattr(pi05, "_PUBLISHED_PARENT_SHA256S", frozenset())
    with pytest.raises(ConfigurationError, match="four published"):
        derive_pi05_draft_from_warm_start(
            warm_start,
            target_contract=_target(),
            initialization_seed=0,
            device="meta",
        )


def test_loss_mask_and_distillation_cover_h50_a32_without_7d_padding() -> None:
    config = Pi05DraftLossConfig(
        executed_prefix_steps=12,
        prefix_weight=1.0,
        tail_weight=0.2,
        padding_weight=0.1,
    )
    mask = build_pi05_draft_loss_mask(
        batch_size=2,
        device="cpu",
        config=config,
    )

    assert mask.shape == (2, 50, 32)
    assert mask[0, 0, 0].item() == pytest.approx(1.0)
    assert mask[0, 0, 31].item() == pytest.approx(0.1)
    assert mask[0, 49, 0].item() == pytest.approx(0.2)
    prediction = torch.zeros((2, 50, 32), requires_grad=True)
    target = torch.ones_like(prediction)
    loss = pi05_draft_distillation_loss(
        prediction,
        target,
        loss_mask=mask,
        config=config,
    )
    assert loss.ndim == 0
    assert loss.item() > 0
    loss.backward()
    assert prediction.grad is not None

    with pytest.raises(ShapeMismatchError, match=r"\[B,50,32\]"):
        pi05_draft_distillation_loss(
            torch.zeros((2, 50, 7)),
            torch.zeros((2, 50, 7)),
        )


def test_candidate_export_binds_lineage_target_training_and_hashes_but_not_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head, target, artifact, metadata = _fake_export(tmp_path, monkeypatch)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert manifest["production_authorized"] is False
    assert manifest["artifact_kind"] == pi05.PI05_DRAFT_ARTIFACT_KIND
    assert manifest["target_contract"] == target.to_dict()
    assert manifest["target_contract_sha256"] == target.sha256
    assert manifest["warm_start"]["checkpoint_sha256"] == (
        head.warm_start_lineage.checkpoint_sha256
    )
    assert manifest["warm_start"]["use_policy"] == ("warm_start_initialization_only")
    assert manifest["training"]["dataset_manifest_sha256"] == "2" * 64
    assert manifest["training"]["optimizer_steps"] == 1
    assert manifest["training"]["metrics"] == {"held_out_weighted_huber": 0.25}
    assert manifest["weights"]["sha256"] == artifact.weights_sha256
    assert manifest["weights"]["state_dict_content_sha256"] == "4" * 64
    assert metadata["production_authorized"] == "false"
    assert metadata["target_contract_sha256"] == target.sha256


def test_untrained_or_mismatched_training_record_cannot_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head, _target_contract = _derived(tmp_path, monkeypatch)
    with pytest.raises(ConfigurationError, match="untrained"):
        export_pi05_draft_candidate(
            head,
            training_record=_training_record(),
            output_dir=tmp_path / "untrained",
        )

    _record_one_step(head)
    with pytest.raises(ConfigurationError, match="optimizer_steps"):
        export_pi05_draft_candidate(
            head,
            training_record=_training_record(steps=2),
            output_dir=tmp_path / "wrong-steps",
        )


def test_ed25519_signed_exact_candidate_loads_frozen_inference_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    candidate_head, target, artifact, metadata = _fake_export(tmp_path, monkeypatch)
    evaluation = _write_evaluation_report(artifact, tmp_path / "evaluation.json")
    private_key = Ed25519PrivateKey.generate()
    private_raw = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    approval = sign_pi05_draft_candidate(
        candidate_manifest_path=artifact.manifest_path,
        evaluation_report_path=evaluation,
        private_key_raw=private_raw,
        signing_key_id="unit-test-key",
        approval_path=tmp_path / "approval.json",
    )

    monkeypatch.setattr(pi05, "_read_safetensors_metadata", lambda _path: metadata)
    monkeypatch.setattr(
        pi05,
        "_load_safetensors",
        lambda _path, _device: candidate_head.state_dict(),
    )
    monkeypatch.setattr(pi05, "_state_dict_content_sha256", lambda _state: "4" * 64)
    adapter = load_production_pi05_draft(
        candidate_manifest_path=artifact.manifest_path,
        approval_path=approval,
        evaluation_report_path=evaluation,
        expected_target=target,
        trusted_public_key_raw=public_raw,
        trusted_signing_key_id="unit-test-key",
        device="meta",
    )

    assert adapter.head.production_compatible is True
    assert adapter.head.training is False
    assert all(not parameter.requires_grad for parameter in adapter.head.parameters())
    observed: dict[str, bool] = {}

    def fake_forward(**inputs: torch.Tensor) -> torch.Tensor:
        observed["inference_mode"] = torch.is_inference_mode_enabled()
        return torch.zeros((inputs["robot_state"].shape[0], 50, 32))

    adapter.head.forward = fake_forward  # type: ignore[method-assign]
    prediction = adapter.predict(
        prefix_embs=torch.zeros((1, 4, 2048)),
        prefix_pad_masks=torch.ones((1, 4), dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 4), dtype=torch.bool),
        robot_state=torch.zeros((1, 32)),
        last_actions=torch.zeros((1, 6, 32)),
    )
    assert observed == {"inference_mode": True}
    assert prediction.actions_model.shape == (1, 50, 32)
    assert prediction.target_contract == target
    assert prediction.artifact_manifest_sha256 == artifact.manifest_sha256

    from types import SimpleNamespace

    from rlinf.projects.fibocom_vla.inference import openpi_speculative
    from rlinf.projects.fibocom_vla.inference.openpi_speculative import (
        OpenPITargetContract,
        VerifiedDraftPrediction,
        VerifiedOpenPITarget,
    )

    main_manifest_path = tmp_path / "main-asset-manifest.json"
    main_norm_path = tmp_path / "norm_stats.json"
    main_manifest_path.write_text("{}", encoding="utf-8")
    main_norm_path.write_text("{}", encoding="utf-8")
    source_fingerprints = tuple(
        SimpleNamespace(sha256_lf=digest) for digest in target.transform_source_sha256
    )
    runtime = SimpleNamespace(
        state_dim=target.raw_state_dim,
        state_semantics=target.state_semantics,
        norm_stats_path=Path(target.norm_stats_path),
        source_fingerprints=source_fingerprints,
    )
    assets = SimpleNamespace(
        manifest=SimpleNamespace(
            repository_id=target.parent_repository_id,
            revision=target.parent_revision,
            runtime=runtime,
        ),
        manifest_path=main_manifest_path,
        norm_stats_path=main_norm_path,
    )
    main_contract = OpenPITargetContract(
        policy_family="pi0.5",
        checkpoint_id="main@revision#weights-sha256:test",
        config_name=target.config_name,
        norm_stats_id=f"{target.asset_id}/{target.norm_stats_path}",
        transform_id=target.transform_id,
        action_horizon=target.action_horizon,
        model_action_dim=target.model_action_dim,
        env_action_indices=target.environment_action_indices,
        state_dim=target.model_state_dim,
        camera_keys=target.raw_camera_keys,
        model_camera_keys=target.model_camera_keys,
        environment_action_semantics=target.environment_action_semantics,
        delta_action_mask=target.delta_action_mask,
        max_token_length=target.max_token_length,
        normalization=target.normalization,
        asset_id=target.asset_id,
    )

    class _MainModel:
        training = False
        config = SimpleNamespace(
            config_name=target.config_name,
            action_horizon=50,
            action_chunk=50,
            action_dim=32,
            action_env_dim=14,
            max_token_len=200,
            num_images_in_input=3,
            pi05=True,
        )

        @staticmethod
        def parameters():
            return ()

    main = object.__new__(VerifiedOpenPITarget)
    object.__setattr__(main, "model", _MainModel())
    object.__setattr__(main, "assets", assets)
    object.__setattr__(main, "contract", main_contract)
    object.__setattr__(main, "_target_token", object())
    object.__setattr__(
        main,
        "_verification_proof",
        openpi_speculative._VERIFIED_OPENPI_TARGET_PROOF,
    )

    def target_file_sha(path: Path) -> str:
        if path == main_manifest_path:
            return target.parent_asset_manifest_sha256
        if path == main_norm_path:
            return target.norm_stats_sha256
        raise AssertionError(path)

    monkeypatch.setattr(pi05, "_sha256_file", target_file_sha)
    assert (
        Pi05DraftTargetContract.from_verified_openpi(
            main,
            draft_parameter_dtype="float32",
        )
        == target
    )
    verified_prediction = adapter.predict_for_target(
        main,
        prefix_embs=torch.zeros((1, 4, 2048)),
        prefix_pad_masks=torch.ones((1, 4), dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 4), dtype=torch.bool),
        robot_state=torch.zeros((1, 32)),
        last_actions=torch.zeros((1, 6, 32)),
    )
    assert isinstance(verified_prediction, VerifiedDraftPrediction)
    assert verified_prediction.target_contract == main_contract
    assert verified_prediction.draft_checkpoint_sha256 == artifact.weights_sha256


def test_unsigned_tampered_or_wrong_target_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _head, target, artifact, _metadata = _fake_export(tmp_path, monkeypatch)
    evaluation = _write_evaluation_report(artifact, tmp_path / "evaluation.json")
    private_key = Ed25519PrivateKey.generate()
    private_raw = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    approval = sign_pi05_draft_candidate(
        candidate_manifest_path=artifact.manifest_path,
        evaluation_report_path=evaluation,
        private_key_raw=private_raw,
        signing_key_id="unit-test-key",
        approval_path=tmp_path / "approval.json",
    )

    with pytest.raises(ConfigurationError, match="failed to read production approval"):
        load_production_pi05_draft(
            candidate_manifest_path=artifact.manifest_path,
            approval_path=tmp_path / "missing-approval.json",
            evaluation_report_path=evaluation,
            expected_target=target,
            trusted_public_key_raw=public_raw,
            trusted_signing_key_id="unit-test-key",
            device="meta",
        )

    with pytest.raises(ConfigurationError, match="differs from loaded main"):
        load_production_pi05_draft(
            candidate_manifest_path=artifact.manifest_path,
            approval_path=approval,
            evaluation_report_path=evaluation,
            expected_target=replace(target, norm_stats_sha256="9" * 64),
            trusted_public_key_raw=public_raw,
            trusted_signing_key_id="unit-test-key",
            device="meta",
        )

    approval_data = json.loads(approval.read_text(encoding="utf-8"))
    approval_data["weights_sha256"] = "a" * 64
    approval.write_text(json.dumps(approval_data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="signature verification failed"):
        load_production_pi05_draft(
            candidate_manifest_path=artifact.manifest_path,
            approval_path=approval,
            evaluation_report_path=evaluation,
            expected_target=target,
            trusted_public_key_raw=public_raw,
            trusted_signing_key_id="unit-test-key",
            device="meta",
        )
