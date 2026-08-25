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

"""Fail-closed policy-space to robot-space policy assembly.

The controller consumes absolute robot-joint chunks, while a checkpoint may
consume a different proprioceptive state and emit calibrated policy-space
actions.  This wrapper keeps that boundary outside both the policy and the
hardware backend.  It also retains model-space action values from the same
forward pass because the adaptation is time preserving.

Custom adapters are loaded from the configured ``module:function`` factory.
The factory receives :class:`RobotConfig` and must return an object with
``observation_to_policy_state`` and ``adapt_action_chunk`` callables.  Native
RTC is disabled for custom adapters unless the object explicitly sets
``rtc_native_safe = True`` and implements
``rtc_conditioning_to_policy(conditioning, observation)``.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import RobotConfig
from ..contracts import (
    ActionChunk,
    ChunkPolicy,
    Observation,
    PolicyOutput,
    RobotState,
)
from ..errors import ConfigurationError, ShapeMismatchError
from ..inference.rtc import RTCConditioning
from .action_adapter import PolicyRobotActionAdapter


def _require_callable(value: Any, *, name: str) -> None:
    if not callable(value):
        raise ConfigurationError(f"policy/robot adapter requires callable {name}")


def load_policy_robot_adapter(robot_config: RobotConfig) -> Any:
    """Load the declared generic or custom policy/robot action adapter.

    Import, lookup, construction, and interface failures are normalized to a
    configuration error so an invalid adapter never degrades to pass-through
    robot commands.
    """

    robot_config.validate()
    implementation = robot_config.action_adapter.implementation
    if implementation == "generic_affine_joint":
        return PolicyRobotActionAdapter(robot_config)

    module_name, separator, factory_name = implementation.partition(":")
    if not separator or not module_name or not factory_name:
        raise ConfigurationError(
            "custom action adapter must use module:function syntax"
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        _require_callable(factory, name=f"factory {implementation}")
        adapter = factory(robot_config)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"failed to load custom action adapter {implementation!r}: {exc}"
        ) from exc

    _require_callable(
        getattr(adapter, "observation_to_policy_state", None),
        name="observation_to_policy_state",
    )
    _require_callable(
        getattr(adapter, "adapt_action_chunk", None),
        name="adapt_action_chunk",
    )
    if bool(getattr(adapter, "rtc_native_safe", False)):
        _require_callable(
            getattr(adapter, "rtc_conditioning_to_policy", None),
            name="rtc_conditioning_to_policy",
        )
    return adapter


class PolicyRobotAdapterPolicy:
    """Adapt a policy's observation and action contracts to one robot backend."""

    def __init__(
        self,
        inner: ChunkPolicy,
        robot_config: RobotConfig,
        *,
        adapter: Any | None = None,
    ) -> None:
        _require_callable(getattr(inner, "predict", None), name="inner.predict")
        self.inner = inner
        self.robot_config = robot_config
        self.adapter = (
            load_policy_robot_adapter(robot_config) if adapter is None else adapter
        )
        _require_callable(
            getattr(self.adapter, "observation_to_policy_state", None),
            name="observation_to_policy_state",
        )
        _require_callable(
            getattr(self.adapter, "adapt_action_chunk", None),
            name="adapt_action_chunk",
        )
        self._generic = isinstance(self.adapter, PolicyRobotActionAdapter)

        has_native_method = callable(getattr(inner, "predict_with_rtc", None))
        inner_native = has_native_method and bool(
            getattr(inner, "rtc_native_supported", has_native_method)
        )
        if self._generic:
            adapter_config = robot_config.action_adapter
            policy_dim = adapter_config.resolved_policy_dim(robot_config.action_dim)
            modes = (
                adapter_config.coordinate_modes
                or (adapter_config.action_mode,) * policy_dim
            )
            adapter_native_safe = all(mode == "absolute" for mode in modes)
        else:
            adapter_native_safe = bool(getattr(self.adapter, "rtc_native_safe", False))
        self.rtc_native_supported = bool(inner_native and adapter_native_safe)

    @staticmethod
    def _policy_observation(
        observation: Observation, policy_state: RobotState
    ) -> Observation:
        if not isinstance(policy_state, RobotState):
            raise ShapeMismatchError(
                "observation_to_policy_state must return RobotState"
            )
        if policy_state.timestamp_ns != observation.state.timestamp_ns:
            raise ShapeMismatchError(
                "adapted policy state must preserve the robot-state timestamp"
            )
        return Observation(
            state=policy_state,
            images=observation.images,
            instruction=observation.instruction,
            timestamp_ns=observation.timestamp_ns,
            frame_id=observation.frame_id,
            metadata=observation.metadata,
        )

    def _adapt_observation(self, observation: Observation) -> Observation:
        policy_state = self.adapter.observation_to_policy_state(observation)
        return self._policy_observation(observation, policy_state)

    def _adapt_action(
        self, action: ActionChunk, observation: Observation
    ) -> ActionChunk:
        adapted = self.adapter.adapt_action_chunk(action, observation)
        if not isinstance(adapted, ActionChunk):
            raise ShapeMismatchError("adapt_action_chunk must return ActionChunk")
        if adapted.action_dim != self.robot_config.action_dim:
            raise ShapeMismatchError(
                "adapted action dimension does not match robot action_dim"
            )
        if adapted.horizon != action.horizon:
            raise ShapeMismatchError(
                "action adaptation must preserve the temporal horizon"
            )
        if not math.isclose(
            adapted.period_s,
            action.period_s,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ShapeMismatchError("action adaptation must preserve period_s")
        if (
            adapted.source_observation_ns != action.source_observation_ns
            or adapted.generated_ns != action.generated_ns
        ):
            raise ShapeMismatchError(
                "action adaptation must preserve source/generated timestamps"
            )
        if adapted.source_observation_ns != observation.timestamp_ns:
            raise ShapeMismatchError(
                "adapted action is not anchored to the supplied observation"
            )

        model_values = (
            None
            if action.model_values is None
            else np.asarray(action.model_values, dtype=np.float32).copy()
        )
        metadata = dict(adapted.metadata)
        metadata["policy_robot_policy_adapter"] = {
            "implementation": self.robot_config.action_adapter.implementation,
            "output_semantics": "absolute_robot_joint_target",
            "model_values_preserved": model_values is not None,
        }
        return ActionChunk(
            values=adapted.values,
            period_s=adapted.period_s,
            source_observation_ns=adapted.source_observation_ns,
            generated_ns=adapted.generated_ns,
            model_values=model_values,
            committed_prefix=adapted.committed_prefix,
            metadata=metadata,
        )

    def _adapt_output(
        self, output: PolicyOutput, observation: Observation
    ) -> PolicyOutput:
        if not isinstance(output, PolicyOutput):
            raise ShapeMismatchError("inner policy must return PolicyOutput")
        action = self._adapt_action(output.action, observation)
        diagnostics = dict(output.diagnostics)
        diagnostics["policy_robot_adapter"] = {
            "implementation": self.robot_config.action_adapter.implementation,
            "rtc_native_supported": self.rtc_native_supported,
        }
        return PolicyOutput(
            action=action,
            model_latency_ms=output.model_latency_ms,
            path=output.path,
            accepted_prefix=output.accepted_prefix,
            diagnostics=diagnostics,
        )

    def predict(self, observation: Observation) -> PolicyOutput:
        """Predict in policy space, then emit absolute robot-joint targets."""

        policy_observation = self._adapt_observation(observation)
        output = self.inner.predict(policy_observation)
        return self._adapt_output(output, observation)

    def _generic_robot_absolute_to_policy(
        self, robot_values: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        values = np.asarray(robot_values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.adapter.robot_dim:
            raise ShapeMismatchError(
                "RTC robot-space targets must be [T, robot_action_dim]"
            )
        if not np.all(np.isfinite(values)):
            raise ShapeMismatchError("RTC robot-space targets must be finite")
        calibrated = (values - self.adapter.offset[None, :]) / self.adapter.scale[
            None, :
        ]
        policy = np.empty((values.shape[0], self.adapter.policy_dim), dtype=np.float32)
        policy[:, self.adapter.robot_from_policy] = calibrated
        return policy

    def _adapt_conditioning(
        self,
        conditioning: RTCConditioning,
        observation: Observation,
    ) -> RTCConditioning:
        if self._generic:
            return RTCConditioning(
                hard_prefix=self._generic_robot_absolute_to_policy(
                    conditioning.hard_prefix
                ),
                overlap_target=self._generic_robot_absolute_to_policy(
                    conditioning.overlap_target
                ),
                overlap_weights=conditioning.overlap_weights,
                model_hard_prefix=conditioning.model_hard_prefix,
                model_overlap_target=conditioning.model_overlap_target,
            )

        converted = self.adapter.rtc_conditioning_to_policy(conditioning, observation)
        if not isinstance(converted, RTCConditioning):
            raise ShapeMismatchError(
                "rtc_conditioning_to_policy must return RTCConditioning"
            )
        if conditioning.model_hard_prefix is None:
            if converted.model_hard_prefix is not None:
                raise ShapeMismatchError(
                    "custom RTC adapter introduced model-space conditioning"
                )
        else:
            if (
                converted.model_hard_prefix is None
                or converted.model_overlap_target is None
                or not np.array_equal(
                    converted.model_hard_prefix,
                    conditioning.model_hard_prefix,
                )
                or not np.array_equal(
                    converted.model_overlap_target,
                    conditioning.model_overlap_target,
                )
            ):
                raise ShapeMismatchError(
                    "custom RTC adapter must preserve model-space conditioning"
                )
        return converted

    def predict_with_rtc(
        self, observation: Observation, conditioning: RTCConditioning
    ) -> PolicyOutput:
        """Run native RTC only when both policy and adapter declare support."""

        if not self.rtc_native_supported:
            raise ConfigurationError(
                "native RTC is not safe across the configured policy/robot adapter"
            )
        policy_observation = self._adapt_observation(observation)
        policy_conditioning = self._adapt_conditioning(conditioning, observation)
        output = self.inner.predict_with_rtc(policy_observation, policy_conditioning)
        return self._adapt_output(output, observation)
