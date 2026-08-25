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

import sys
import types

import pytest

from rlinf.projects.fibocom_vla.config import (
    ResidualRLConfig,
    SpeculativeConfig,
    StackConfig,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError
from rlinf.projects.fibocom_vla.factories import create_openpi_policy_from_env
from rlinf.projects.fibocom_vla.openpi_adapter import RLinfOpenPiChunkPolicy


class _FakeModel:
    def __init__(
        self,
        *,
        action_horizon: int = 50,
        action_chunk: int = 50,
        action_env_dim: int = 7,
        action_dim: int = 32,
    ) -> None:
        self.config = types.SimpleNamespace(
            action_horizon=action_horizon,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
            action_dim=action_dim,
        )

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def parameters(self):
        return ()


def _base_only_config() -> StackConfig:
    return StackConfig(
        residual_rl=ResidualRLConfig(enabled=False),
        speculative=SpeculativeConfig(enabled=False),
    )


def _patch_model_loader(monkeypatch, **model_kwargs) -> _FakeModel:
    model = _FakeModel(**model_kwargs)
    models = types.ModuleType("rlinf.models")
    models.__path__ = []
    embodiment = types.ModuleType("rlinf.models.embodiment")
    embodiment.__path__ = []
    openpi = types.ModuleType("rlinf.models.embodiment.openpi")
    openpi.get_model = lambda config: model
    monkeypatch.setitem(sys.modules, "rlinf.models", models)
    monkeypatch.setitem(sys.modules, "rlinf.models.embodiment", embodiment)
    monkeypatch.setitem(sys.modules, "rlinf.models.embodiment.openpi", openpi)
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
    monkeypatch.delenv("FIBOCOM_DRAFT_POLICY_FACTORY", raising=False)
    with pytest.raises(ConfigurationError, match="DRAFT_POLICY_FACTORY"):
        create_openpi_policy_from_env(config)
