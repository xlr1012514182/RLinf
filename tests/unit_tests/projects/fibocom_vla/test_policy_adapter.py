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
from dataclasses import replace

import numpy as np
import pytest

from rlinf.projects.fibocom_vla.config import ActionAdapterConfig, RobotConfig
from rlinf.projects.fibocom_vla.contracts import (
    ActionChunk,
    Observation,
    PolicyOutput,
    RobotState,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError
from rlinf.projects.fibocom_vla.hardware.policy_adapter import (
    PolicyRobotAdapterPolicy,
)
from rlinf.projects.fibocom_vla.inference.rtc import RTCConditioning


def _adapter_config(
    *,
    mode: str = "absolute",
    coordinate_modes: tuple[str, ...] = (),
    implementation: str = "generic_affine_joint",
) -> ActionAdapterConfig:
    return ActionAdapterConfig(
        implementation=implementation,
        action_mode=mode,
        coordinate_modes=coordinate_modes,
        policy_frame="joint",
        policy_dim=2,
        policy_names=("p0", "p1"),
        policy_units=("radian", "radian"),
        policy_state_frame="joint",
        policy_state_dim=2,
        policy_state_names=("p0", "p1"),
        required_cameras=("main", "wrist"),
        robot_units=("degree", "degree"),
        robot_from_policy=(1, 0),
        scale=(2.0, -4.0),
        offset=(10.0, 20.0),
        validated=False,
    )


def _robot_config(
    *,
    mode: str = "absolute",
    coordinate_modes: tuple[str, ...] = (),
    implementation: str = "generic_affine_joint",
) -> RobotConfig:
    return RobotConfig(
        action_dim=2,
        control_hz=10.0,
        joint_names=("r0", "r1"),
        joint_lower=(-100.0, -100.0),
        joint_upper=(100.0, 100.0),
        max_step=(100.0, 100.0),
        dry_run=True,
        action_adapter=_adapter_config(
            mode=mode,
            coordinate_modes=coordinate_modes,
            implementation=implementation,
        ),
    )


def _observation() -> Observation:
    main = np.zeros((2, 3, 3), dtype=np.uint8)
    wrist = np.ones((2, 3, 3), dtype=np.uint8)
    return Observation(
        state=RobotState(
            joint_positions=np.asarray([14.0, 16.0], dtype=np.float32),
            joint_names=("r0", "r1"),
            timestamp_ns=990,
            joint_velocities=np.asarray([6.0, -8.0], dtype=np.float32),
        ),
        images={"main": main, "wrist": wrist},
        instruction="stack the blocks",
        timestamp_ns=1_000,
        frame_id=17,
        metadata={"clock_domain": "monotonic"},
    )


def _policy_output(observation: Observation) -> PolicyOutput:
    return PolicyOutput(
        action=ActionChunk(
            values=np.asarray([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32),
            period_s=0.1,
            source_observation_ns=observation.timestamp_ns,
            generated_ns=observation.timestamp_ns + 1,
            model_values=np.arange(8, dtype=np.float32).reshape(2, 4),
            committed_prefix=1,
            metadata={"inner": True},
        ),
        model_latency_ms=2.5,
        path="unit-test",
        accepted_prefix=1,
        diagnostics={"forward_count": 1},
    )


class _RecordingPolicy:
    def __init__(self) -> None:
        self.observation: Observation | None = None

    def predict(self, observation: Observation) -> PolicyOutput:
        self.observation = observation
        return _policy_output(observation)


class _NativeRecordingPolicy(_RecordingPolicy):
    rtc_native_supported = True

    def __init__(self) -> None:
        super().__init__()
        self.conditioning: RTCConditioning | None = None

    def predict_with_rtc(
        self, observation: Observation, conditioning: RTCConditioning
    ) -> PolicyOutput:
        self.observation = observation
        self.conditioning = conditioning
        return _policy_output(observation)


def _conditioning() -> RTCConditioning:
    return RTCConditioning(
        hard_prefix=np.asarray([[18.0, 8.0]], dtype=np.float32),
        overlap_target=np.asarray([[18.0, 8.0], [20.0, 4.0]], dtype=np.float32),
        overlap_weights=np.asarray([1.0, 0.5], dtype=np.float32),
        model_hard_prefix=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        model_overlap_target=np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        ),
    )


def test_nonidentity_policy_wrapper_adapts_state_action_and_provenance() -> None:
    inner = _RecordingPolicy()
    wrapper = PolicyRobotAdapterPolicy(inner, _robot_config())
    observation = _observation()

    output = wrapper.predict(observation)

    assert inner.observation is not None
    np.testing.assert_allclose(inner.observation.state.joint_positions, [1.0, 2.0])
    np.testing.assert_allclose(inner.observation.state.joint_velocities, [2.0, 3.0])
    assert inner.observation.state.joint_names == ("p0", "p1")
    assert inner.observation.images["main"] is observation.images["main"]
    assert inner.observation.images["wrist"] is observation.images["wrist"]
    assert inner.observation.instruction == observation.instruction
    assert inner.observation.timestamp_ns == observation.timestamp_ns
    assert inner.observation.frame_id == observation.frame_id
    assert inner.observation.metadata == observation.metadata

    np.testing.assert_allclose(output.action.values, [[18.0, 8.0], [20.0, 4.0]])
    np.testing.assert_array_equal(
        output.action.model_values,
        np.arange(8, dtype=np.float32).reshape(2, 4),
    )
    assert output.action.source_observation_ns == observation.timestamp_ns
    assert output.action.generated_ns == observation.timestamp_ns + 1
    assert output.action.committed_prefix == 1
    assert output.action.metadata["policy_robot_policy_adapter"][
        "model_values_preserved"
    ]
    assert output.path == "unit-test"
    assert output.accepted_prefix == 1


def test_generic_absolute_native_rtc_inverts_environment_targets() -> None:
    inner = _NativeRecordingPolicy()
    wrapper = PolicyRobotAdapterPolicy(inner, _robot_config())
    conditioning = _conditioning()

    output = wrapper.predict_with_rtc(_observation(), conditioning)

    assert wrapper.rtc_native_supported is True
    assert inner.conditioning is not None
    np.testing.assert_allclose(inner.conditioning.hard_prefix, [[3.0, 4.0]])
    np.testing.assert_allclose(
        inner.conditioning.overlap_target, [[3.0, 4.0], [4.0, 5.0]]
    )
    np.testing.assert_array_equal(
        inner.conditioning.model_hard_prefix,
        conditioning.model_hard_prefix,
    )
    np.testing.assert_array_equal(
        inner.conditioning.model_overlap_target,
        conditioning.model_overlap_target,
    )
    np.testing.assert_allclose(output.action.values, [[18.0, 8.0], [20.0, 4.0]])
    np.testing.assert_array_equal(
        output.action.model_values,
        np.arange(8, dtype=np.float32).reshape(2, 4),
    )


@pytest.mark.parametrize(
    ("mode", "coordinate_modes"),
    [
        ("delta_from_observation", ()),
        ("absolute", ("absolute", "velocity")),
    ],
)
def test_generic_nonabsolute_contract_disables_native_rtc(
    mode: str, coordinate_modes: tuple[str, ...]
) -> None:
    wrapper = PolicyRobotAdapterPolicy(
        _NativeRecordingPolicy(),
        _robot_config(mode=mode, coordinate_modes=coordinate_modes),
    )

    assert wrapper.rtc_native_supported is False
    with pytest.raises(ConfigurationError, match="native RTC is not safe"):
        wrapper.predict_with_rtc(_observation(), _conditioning())


def _install_custom_module(monkeypatch, name: str, factory) -> None:
    module = types.ModuleType(name)
    module.create_adapter = factory
    monkeypatch.setitem(sys.modules, name, module)


class _CustomAdapter:
    def observation_to_policy_state(self, observation: Observation) -> RobotState:
        return observation.state

    def adapt_action_chunk(
        self, chunk: ActionChunk, observation: Observation
    ) -> ActionChunk:
        del observation
        return replace(chunk, model_values=None)


def test_custom_factory_is_loaded_but_native_rtc_defaults_closed(
    monkeypatch,
) -> None:
    _install_custom_module(
        monkeypatch,
        "fibocom_test_custom_closed",
        lambda config: _CustomAdapter(),
    )
    config = _robot_config(implementation="fibocom_test_custom_closed:create_adapter")

    wrapper = PolicyRobotAdapterPolicy(_NativeRecordingPolicy(), config)
    output = wrapper.predict(_observation())

    assert wrapper.rtc_native_supported is False
    assert output.action.model_values is not None


def test_custom_factory_must_supply_declared_interface(monkeypatch) -> None:
    _install_custom_module(
        monkeypatch,
        "fibocom_test_custom_invalid",
        lambda config: object(),
    )
    config = _robot_config(implementation="fibocom_test_custom_invalid:create_adapter")

    with pytest.raises(ConfigurationError, match="observation_to_policy_state"):
        PolicyRobotAdapterPolicy(_RecordingPolicy(), config)


def test_custom_native_declaration_requires_converter(monkeypatch) -> None:
    class UnsafeDeclaration(_CustomAdapter):
        rtc_native_safe = True

    _install_custom_module(
        monkeypatch,
        "fibocom_test_custom_missing_rtc",
        lambda config: UnsafeDeclaration(),
    )
    config = _robot_config(
        implementation="fibocom_test_custom_missing_rtc:create_adapter"
    )

    with pytest.raises(ConfigurationError, match="rtc_conditioning_to_policy"):
        PolicyRobotAdapterPolicy(_NativeRecordingPolicy(), config)


def test_custom_native_converter_preserves_model_conditioning(monkeypatch) -> None:
    class NativeCustomAdapter(_CustomAdapter):
        rtc_native_safe = True

        def __init__(self) -> None:
            self.observation: Observation | None = None

        def rtc_conditioning_to_policy(
            self,
            conditioning: RTCConditioning,
            observation: Observation,
        ) -> RTCConditioning:
            self.observation = observation
            return conditioning

    custom = NativeCustomAdapter()
    _install_custom_module(
        monkeypatch,
        "fibocom_test_custom_native",
        lambda config: custom,
    )
    config = _robot_config(implementation="fibocom_test_custom_native:create_adapter")

    wrapper = PolicyRobotAdapterPolicy(_NativeRecordingPolicy(), config)
    conditioning = _conditioning()
    output = wrapper.predict_with_rtc(_observation(), conditioning)

    assert wrapper.rtc_native_supported is True
    assert custom.observation is not None
    np.testing.assert_array_equal(
        output.action.model_values,
        np.arange(8, dtype=np.float32).reshape(2, 4),
    )
