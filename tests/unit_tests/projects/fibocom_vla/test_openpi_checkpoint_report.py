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

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rlinf.projects.fibocom_vla import cli
from rlinf.projects.fibocom_vla.contracts import ActionChunk, PolicyOutput

torch = pytest.importorskip("torch")

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "rlinf"
    / "models"
    / "embodiment"
    / "openpi"
    / "__init__.py"
)
_SPEC = importlib.util.spec_from_file_location("_fibocom_openpi_loader", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
CHECKPOINT_LOAD_REPORT_ATTR = _MODULE.CHECKPOINT_LOAD_REPORT_ATTR
_load_state_dict_with_report = _MODULE._load_state_dict_with_report


def test_checkpoint_load_report_records_exact_keys_and_selected_paths(
    tmp_path: Path,
):
    model = torch.nn.Linear(2, 1)
    selected = tmp_path / "model.safetensors"

    report = _load_state_dict_with_report(
        model,
        model.state_dict(),
        source_kind="safetensors_shards",
        selected_paths=(selected,),
    )

    assert report == getattr(model, CHECKPOINT_LOAD_REPORT_ATTR)
    assert report["source_kind"] == "safetensors_shards"
    assert report["selected_paths"] == (str(selected.resolve()),)
    assert report["missing_keys"] == ()
    assert report["unexpected_keys"] == ()


def test_checkpoint_load_report_preserves_every_incompatible_key(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    state_dict = {"weight": model.weight.detach().clone(), "extra": torch.zeros(1)}

    report = _load_state_dict_with_report(
        model,
        state_dict,
        source_kind="model_state_dict_full_weights",
        selected_paths=(tmp_path / "full_weights.pt",),
    )

    assert report["missing_keys"] == ("bias",)
    assert report["unexpected_keys"] == ("extra",)


def test_checkpoint_smoke_reports_h50_model_and_environment_shapes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = torch.nn.Linear(1, 1)
    model._rlinf_checkpoint_load_report = {
        "source_kind": "safetensors_shards",
        "selected_paths": ("/verified/model.safetensors",),
        "missing_keys": (),
        "unexpected_keys": (),
        "data_asset_id": "physical-intelligence/robotwin",
        "use_quantile_norm": True,
    }

    class _Policy:
        fibocom_checkpoint_identity = {
            "repository_id": "RLinf/test",
            "action_horizon": 50,
            "model_action_dim": 32,
        }
        fibocom_stack_layers = ("rlinf_openpi_pi05",)

        def __init__(self) -> None:
            self.model = model

        @staticmethod
        def predict(observation) -> PolicyOutput:
            return PolicyOutput(
                action=ActionChunk(
                    values=np.zeros((50, 14), dtype=np.float32),
                    model_values=np.zeros((50, 32), dtype=np.float32),
                    period_s=0.05,
                    source_observation_ns=observation.timestamp_ns,
                ),
                model_latency_ms=1.0,
                path="unit_openpi",
            )

    import rlinf.projects.fibocom_vla.factories as factories

    monkeypatch.setattr(
        factories,
        "create_openpi_policy_from_env",
        lambda _config: _Policy(),
    )
    config = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "embodiment"
        / "fibocom_vla"
        / "config"
        / "robotwin_pi05_h50_dry_run.json"
    )

    assert (
        cli._openpi_checkpoint_smoke(
            argparse.Namespace(
                config=config,
                instruction="stack the block",
                seed=17,
            )
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["smoke_passed"] is True
    assert report["environment_action_shape"] == [50, 14]
    assert report["model_action_shape"] == [50, 32]
    assert report["all_outputs_finite"] is True
