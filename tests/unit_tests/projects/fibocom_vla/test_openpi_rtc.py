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

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rlinf.projects.fibocom_vla.errors import ShapeMismatchError  # noqa: E402
from rlinf.projects.fibocom_vla.inference.openpi_rtc import (  # noqa: E402
    OPENPI_RTC_DIAGNOSTICS_KEY,
    OPENPI_RTC_PREFIX_FEATURE_KEY,
    OpenPINativeRTCAdapter,
    OpenPIRTCContext,
    masked_mean_prefix_feature,
)


class _FakeReleaseV02OpenPI(torch.nn.Module):
    """Small differentiable stand-in for the release/v0.2 OpenPI surface."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            action_horizon=3,
            action_dim=2,
            num_steps=2,
            action_chunk=3,
            action_env_dim=2,
            add_value_head=False,
            value_after_vlm=False,
            chunk_critic_input=False,
        )
        self.action_in_proj = torch.nn.Linear(2, 2, bias=False)
        self.action_out_proj = torch.nn.Identity()
        self.use_vlm_value = False
        self.prefix_calls = 0
        self.denoiser_inputs: list[torch.Tensor] = []
        self.denoiser_requires_grad: list[bool] = []

    def _preprocess_observation(self, observation, *, train):
        assert train is False
        state = observation.state
        return [state], [None], None, None, state

    def _build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks):
        del img_masks, lang_tokens, lang_masks
        self.prefix_calls += 1
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
        return 0.5 * x_t

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device, dtype=self.action_in_proj.weight.dtype)


class _FakeReleaseV02InlinePrefix(_FakeReleaseV02OpenPI):
    """Expose only the prefix primitives available in release/v0.2."""

    _build_prefix_cache = None

    def __init__(self) -> None:
        super().__init__()
        language_config = SimpleNamespace(_attn_implementation=None)
        paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=language_config)
        )
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=paligemma,
            forward=self._prefix_forward,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        del img_masks, lang_tokens, lang_masks
        batch_size = images[0].shape[0]
        embeddings = torch.zeros((batch_size, 3, 4), device=images[0].device)
        pad_masks = torch.tensor([True, False, True], device=images[0].device).expand(
            batch_size, 3
        )
        attention_masks = torch.zeros_like(pad_masks)
        return embeddings, pad_masks, attention_masks

    def _prepare_attention_masks_4d(self, masks):
        return masks[:, None]

    def _prefix_forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        self.prefix_calls += 1
        batch_size = inputs_embeds[0].shape[0]
        token_values = torch.tensor(
            [1.0, 99.0, 3.0], device=inputs_embeds[0].device
        ).view(1, 3, 1)
        prefix_output = token_values.expand(batch_size, 3, 2048).clone()
        return (prefix_output, None), object()


def _observation(batch_size: int = 1):
    return SimpleNamespace(state=torch.zeros((batch_size, 1), dtype=torch.float32))


def test_native_vjp_hard_freeze_and_single_prefix_feature_forward():
    model = _FakeReleaseV02OpenPI()
    previous = torch.tensor([[[0.25, -0.5], [1.0, 1.0], [1.0, 1.0]]])
    context = OpenPIRTCContext(
        previous,
        hard_prefix_steps=1,
        guidance_scale=1.0,
        guidance_decay=0.5,
    )

    result = OpenPINativeRTCAdapter(model).sample_actions(
        _observation(),
        context,
        noise=torch.zeros((1, 3, 2)),
    )

    assert model.prefix_calls == 1
    feature = result[OPENPI_RTC_PREFIX_FEATURE_KEY]
    assert feature.shape == (1, 2048)
    torch.testing.assert_close(feature, torch.full_like(feature, 2.0))

    expected_hard = previous[:, :1]
    for denoiser_input in model.denoiser_inputs:
        torch.testing.assert_close(denoiser_input[:, :1], expected_hard)
    assert model.denoiser_requires_grad == [True, True]
    torch.testing.assert_close(result["actions"][:, :1], expected_hard)
    torch.testing.assert_close(
        result["chains"][:, :, :1],
        expected_hard[:, None].expand(-1, 3, -1, -1),
    )

    # The earlier soft target gets the larger exponential position weight.
    action_magnitude = result["actions"].abs().sum(dim=-1)
    assert action_magnitude[0, 1] > action_magnitude[0, 2] > 0
    assert all(parameter.grad is None for parameter in model.parameters())

    diagnostics = result[OPENPI_RTC_DIAGNOSTICS_KEY]
    assert diagnostics["rtc_native_guidance"] is True
    assert diagnostics["rtc_host_postprocess_fallback"] is False
    assert diagnostics["vjp_steps"] == 2
    assert diagnostics["prefix_cache_path"] == "model._build_prefix_cache"
    assert diagnostics["velocity_paths"] == (
        "release_v0.2.get_suffix_out+action_out_proj",
    )
    assert diagnostics["prefix_feature_reused_same_forward"] is True
    assert diagnostics["prefix_feature_dim"] == 2048
    assert diagnostics["real_checkpoint_validated"] is False
    assert diagnostics["gpu_validated"] is False


def test_no_remaining_overlap_matches_release_v02_euler_flow():
    model = _FakeReleaseV02OpenPI()
    context = OpenPIRTCContext(
        torch.zeros((1, 3, 2)),
        executed_steps=3,
        guidance_scale=1.0,
    )

    result = OpenPINativeRTCAdapter(model).sample_actions(
        _observation(),
        context,
        noise=torch.ones((1, 3, 2)),
    )

    # v=0.5*x, two dt=-0.5 Euler updates: 1 * 0.75 * 0.75.
    torch.testing.assert_close(result["actions"], torch.full((1, 3, 2), 0.5625))
    diagnostics = result[OPENPI_RTC_DIAGNOSTICS_KEY]
    assert diagnostics["vjp_steps"] == 0
    assert diagnostics["overlap_lengths"] == (0,)
    assert model.denoiser_requires_grad == [False, False]
    assert model.prefix_calls == 1


def test_release_v02_inline_prefix_path_reuses_its_only_forward(monkeypatch):
    openpi_module = ModuleType("openpi")
    openpi_module.__path__ = []
    models_module = ModuleType("openpi.models_pytorch")
    models_module.__path__ = []
    pi0_module = ModuleType("openpi.models_pytorch.pi0_pytorch")
    pi0_module.make_att_2d_masks = lambda pad_masks, attention_masks: (
        pad_masks[:, None, :] & ~attention_masks[:, :, None]
    )
    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.setitem(sys.modules, "openpi.models_pytorch", models_module)
    monkeypatch.setitem(
        sys.modules,
        "openpi.models_pytorch.pi0_pytorch",
        pi0_module,
    )
    model = _FakeReleaseV02InlinePrefix()
    context = OpenPIRTCContext(torch.zeros((1, 3, 2)), executed_steps=3)

    result = OpenPINativeRTCAdapter(model).sample_actions(
        _observation(),
        context,
        noise=torch.ones((1, 3, 2)),
    )

    assert model.prefix_calls == 1
    torch.testing.assert_close(
        result[OPENPI_RTC_PREFIX_FEATURE_KEY],
        torch.full((1, 2048), 2.0),
    )
    diagnostics = result[OPENPI_RTC_DIAGNOSTICS_KEY]
    assert diagnostics["prefix_cache_path"] == "release_v0.2_inline_prefix_cache"
    assert diagnostics["prefix_feature_reused_same_forward"] is True


def test_per_batch_alignment_and_hard_lengths():
    model = _FakeReleaseV02OpenPI()
    previous = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    context = OpenPIRTCContext(
        previous,
        executed_steps=[1, 0],
        hard_prefix_steps=[2, 1],
        overlap_horizon=2,
        guidance_scale=0.0,
    )

    result = OpenPINativeRTCAdapter(model).sample_actions(
        _observation(2),
        context,
        noise=torch.zeros((2, 3, 2)),
    )

    torch.testing.assert_close(result["actions"][0, :2], previous[0, 1:3])
    torch.testing.assert_close(result["actions"][1, :1], previous[1, :1])
    diagnostics = result[OPENPI_RTC_DIAGNOSTICS_KEY]
    assert diagnostics["executed_steps"] == (1, 0)
    assert diagnostics["hard_prefix_lengths"] == (2, 1)
    assert diagnostics["overlap_lengths"] == (2, 2)


def test_inference_mode_is_rejected_because_native_rtc_needs_autograd():
    model = _FakeReleaseV02OpenPI()
    context = OpenPIRTCContext(torch.zeros((1, 3, 2)))

    with torch.inference_mode(), pytest.raises(RuntimeError, match="needs autograd"):
        OpenPINativeRTCAdapter(model).sample_actions(_observation(), context)


def test_masked_mean_prefix_feature_checks_pi05_width_and_valid_tokens():
    output = torch.ones((1, 2, 16))
    with pytest.raises(ShapeMismatchError, match="2048-D"):
        masked_mean_prefix_feature(output, torch.ones((1, 2), dtype=torch.bool))
    with pytest.raises(ShapeMismatchError, match="valid token"):
        masked_mean_prefix_feature(
            torch.ones((1, 2, 2048)),
            torch.zeros((1, 2), dtype=torch.bool),
        )
