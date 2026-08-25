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

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlinf.projects.fibocom_vla.contracts import (  # noqa: E402
    Observation,
    RobotState,
)
from rlinf.projects.fibocom_vla.errors import ShapeMismatchError  # noqa: E402
from rlinf.projects.fibocom_vla.inference.rtc import RTCConditioning  # noqa: E402
from rlinf.projects.fibocom_vla.openpi_adapter import (  # noqa: E402
    OpenPiAdapterConfig,
    RLinfOpenPiChunkPolicy,
)
from rlinf.projects.fibocom_vla.residual_policy import (  # noqa: E402
    FrozenReferenceOutput,
)


class _FakeOpenPI(torch.nn.Module):
    """Release/v0.2-shaped model with observable prefix and transform calls."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            action_horizon=3,
            action_dim=2,
            num_steps=2,
            action_chunk=2,
            action_env_dim=2,
            add_value_head=False,
            value_after_vlm=False,
            chunk_critic_input=False,
        )
        self.action_in_proj = torch.nn.Linear(2, 2, bias=False)
        self.action_out_proj = torch.nn.Identity()
        self.use_vlm_value = False
        self.prefix_calls = 0
        self.prefix_inference_modes: list[bool] = []
        self.denoiser_inputs: list[torch.Tensor] = []
        self.denoiser_requires_grad: list[bool] = []
        self.output_transform_calls = 0

    def obs_processor(self, env_observation):
        return {
            "observation/image": env_observation["main_images"],
            "observation/state": env_observation["states"],
            "prompt": env_observation["task_descriptions"],
        }

    def input_transform(self, observation, *, transpose):
        assert transpose is False
        return {
            "state": observation["observation/state"].to(dtype=torch.float32),
            "image": observation["observation/image"],
            "tokenized_prompt": torch.ones((1, 2), dtype=torch.long),
            "tokenized_prompt_mask": torch.ones((1, 2), dtype=torch.bool),
        }

    def precision_processor(self, observation):
        return observation

    def observation_from_dict(self, observation):
        return SimpleNamespace(state=observation["state"], payload=observation)

    def _preprocess_observation(self, observation, *, train):
        assert train is False
        return [observation.payload["image"]], [None], None, None, observation.state

    def _build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks):
        del img_masks, lang_tokens, lang_masks
        self.prefix_calls += 1
        self.prefix_inference_modes.append(torch.is_inference_mode_enabled())
        batch_size = images[0].shape[0]
        token_values = torch.tensor([1.0, 99.0, 3.0], device=images[0].device).view(
            1, 3, 1
        )
        prefix_output = token_values.expand(batch_size, 3, 2048).clone()
        prefix_pad_masks = torch.tensor(
            [True, False, True], device=images[0].device
        ).expand(batch_size, 3)
        return prefix_output, prefix_pad_masks, object()

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        del state, prefix_pad_masks, past_key_values, timestep
        self.denoiser_inputs.append(x_t.detach().clone())
        self.denoiser_requires_grad.append(x_t.requires_grad)
        scale = torch.tensor([0.5, 0.25], device=x_t.device, dtype=x_t.dtype)
        return x_t * scale

    def sample_noise(self, shape, device):
        return torch.ones(shape, device=device, dtype=self.action_in_proj.weight.dtype)

    def output_transform(self, outputs):
        self.output_transform_calls += 1
        state_offset = outputs["state"][:, None, :2]
        return {"actions": (outputs["actions"] * 10.0 + 1.0 + state_offset)[:, :2]}


def _observation(
    positions: tuple[float, float] = (0.1, -0.2),
) -> Observation:
    return Observation(
        state=RobotState(
            np.asarray(positions, dtype=np.float32),
            ("joint_1", "joint_2"),
        ),
        images={"main": np.zeros((4, 4, 3), dtype=np.uint8)},
        instruction="stack the blocks",
    )


def _policy(model: _FakeOpenPI) -> RLinfOpenPiChunkPolicy:
    return RLinfOpenPiChunkPolicy(
        model,
        OpenPiAdapterConfig(period_s=0.05, action_indices=(1, 0)),
    )


def test_predict_with_features_reuses_one_prefix_forward_and_preserves_raw_chunk():
    model = _FakeOpenPI()
    policy = _policy(model)

    frozen = policy.predict_with_features(_observation())

    assert isinstance(frozen, FrozenReferenceOutput)
    assert model.prefix_calls == 1
    assert model.output_transform_calls == 1
    assert model.prefix_inference_modes == [True]
    assert frozen.features.shape == (2048,)
    np.testing.assert_allclose(frozen.features, 2.0)

    # Two release/v0.2 Euler steps from noise=1.0.
    expected_model = np.asarray([0.5625, 0.765625], dtype=np.float32)
    expected_env = (
        expected_model * 10.0 + 1.0 + np.asarray([0.1, -0.2], dtype=np.float32)
    )[[1, 0]]
    np.testing.assert_allclose(
        frozen.output.action.model_values,
        np.tile(expected_model, (2, 1)),
    )
    np.testing.assert_allclose(
        frozen.output.action.values,
        np.tile(expected_env, (2, 1)),
    )
    assert frozen.output.action.metadata["openpi_model_values_preserved"] is True
    assert frozen.output.diagnostics["prefix_feature_reused_same_forward"] is True
    assert frozen.output.diagnostics["prefix_feature_dim"] == 2048
    assert frozen.output.diagnostics["rtc_native_guidance"] is False


def test_predict_delegates_to_same_feature_path_without_a_second_forward():
    model = _FakeOpenPI()

    output = _policy(model).predict(_observation())

    assert model.prefix_calls == 1
    assert output.action.model_values is not None
    assert output.path == "rlinf_openpi_native"


def test_predict_with_rtc_uses_model_lineage_outside_inference_mode():
    model = _FakeOpenPI()
    policy = _policy(model)
    previous = policy.predict(_observation()).action
    model.prefix_calls = 0
    model.prefix_inference_modes.clear()
    model.denoiser_inputs.clear()
    model.denoiser_requires_grad.clear()
    hard_length = 1
    overlap = 2
    conditioning = RTCConditioning(
        hard_prefix=previous.values[:hard_length],
        overlap_target=previous.values[:overlap],
        overlap_weights=np.asarray([1.0, 0.5], dtype=np.float32),
        model_hard_prefix=previous.model_values[:hard_length],
        model_overlap_target=previous.model_values[:overlap],
    )

    output = policy.predict_with_rtc(_observation((0.7, 0.6)), conditioning)

    assert model.prefix_calls == 1
    assert model.prefix_inference_modes == [False]
    assert model.denoiser_requires_grad == [True, True]
    for denoiser_input in model.denoiser_inputs:
        torch.testing.assert_close(
            denoiser_input[:, :hard_length],
            torch.from_numpy(previous.model_values[None, :hard_length]),
        )
    np.testing.assert_allclose(
        output.action.model_values[:hard_length],
        previous.model_values[:hard_length],
    )
    np.testing.assert_allclose(
        output.action.values[:hard_length],
        previous.values[:hard_length],
    )
    assert output.action.committed_prefix == hard_length
    assert output.path == "rlinf_openpi_native_rtc_vjp"
    assert output.diagnostics["rtc_native_guidance"] is True
    assert output.diagnostics["rtc_host_postprocess_fallback"] is False
    assert output.diagnostics["committed_env_prefix_reused"] is True
    assert output.diagnostics["overlap_weight_source"] == "explicit"
    assert output.diagnostics["soft_position_weights"] == ((0.0, 0.5),)
    assert output.diagnostics["vjp_steps"] == 2


def test_predict_with_rtc_fails_closed_without_model_space_lineage():
    model = _FakeOpenPI()
    conditioning = RTCConditioning(
        hard_prefix=np.zeros((1, 2), dtype=np.float32),
        overlap_target=np.zeros((2, 2), dtype=np.float32),
        overlap_weights=np.asarray([1.0, 0.5], dtype=np.float32),
    )

    with pytest.raises(ShapeMismatchError, match="model-space lineage"):
        _policy(model).predict_with_rtc(_observation(), conditioning)

    assert model.prefix_calls == 0


def test_aloha_adapter_preserves_ordered_two_wrist_camera_axis():
    model = _FakeOpenPI()
    policy = RLinfOpenPiChunkPolicy(
        model,
        OpenPiAdapterConfig(
            main_camera="cam_high",
            wrist_cameras=("cam_left_wrist", "cam_right_wrist"),
            expected_state_names=("joint_1", "joint_2"),
        ),
    )
    observation = Observation(
        state=RobotState(
            np.zeros(2, dtype=np.float32),
            ("joint_1", "joint_2"),
        ),
        images={
            "cam_high": np.full((4, 5, 3), 1, dtype=np.uint8),
            "cam_left_wrist": np.full((4, 5, 3), 2, dtype=np.uint8),
            "cam_right_wrist": np.full((4, 5, 3), 3, dtype=np.uint8),
        },
        instruction="stack the blocks",
    )

    env_observation = policy._to_env_observation(observation)

    assert tuple(env_observation["main_images"].shape) == (1, 4, 5, 3)
    assert tuple(env_observation["wrist_images"].shape) == (1, 2, 4, 5, 3)
    assert torch.all(env_observation["wrist_images"][:, 0] == 2)
    assert torch.all(env_observation["wrist_images"][:, 1] == 3)
    assert env_observation["extra_view_images"] is None


def test_adapter_checks_checkpoint_state_names_on_every_observation():
    policy = RLinfOpenPiChunkPolicy(
        _FakeOpenPI(),
        OpenPiAdapterConfig(expected_state_names=("joint_1", "joint_2")),
    )

    policy._to_env_observation(_observation())
    renamed = Observation(
        state=RobotState(
            np.zeros(2, dtype=np.float32),
            ("joint_2", "joint_1"),
        ),
        images={"main": np.zeros((4, 4, 3), dtype=np.uint8)},
        instruction="stack the blocks",
    )
    with pytest.raises(ShapeMismatchError, match="joint_names"):
        policy._to_env_observation(renamed)


def test_adapter_rejects_ambiguous_or_missing_wrist_camera_mapping():
    with pytest.raises(ValueError, match="not both"):
        OpenPiAdapterConfig(
            wrist_camera="wrist",
            wrist_cameras=("left", "right"),
        )

    policy = RLinfOpenPiChunkPolicy(
        _FakeOpenPI(),
        OpenPiAdapterConfig(wrist_cameras=("left", "right")),
    )
    with pytest.raises(ShapeMismatchError, match="'left', 'right'"):
        policy._to_env_observation(_observation())
