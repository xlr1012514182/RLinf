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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rlinf.projects.fibocom_vla.errors import (  # noqa: E402
    ConfigurationError,
    ShapeMismatchError,
)
from rlinf.projects.fibocom_vla.inference import (  # noqa: E402
    openpi_speculative as speculative_module,
)
from rlinf.projects.fibocom_vla.inference.openpi_speculative import (  # noqa: E402
    OpenPIParallelVerifier,
    OpenPITargetContract,
    VerifiedOpenPITarget,
    _issue_verified_draft_prediction,
    _require_exact_model_load,
    triton_postprocess_batch,
)
from rlinf.projects.fibocom_vla.inference.triton_speculative import (  # noqa: E402
    TritonSpeculativeConfig,
)

_OPENPI_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "rlinf"
    / "models"
    / "embodiment"
    / "openpi"
    / "__init__.py"
)
_OPENPI_SPEC = importlib.util.spec_from_file_location(
    "_fibocom_speculative_openpi_loader", _OPENPI_SOURCE
)
assert _OPENPI_SPEC is not None and _OPENPI_SPEC.loader is not None
_OPENPI_LOADER = importlib.util.module_from_spec(_OPENPI_SPEC)
_OPENPI_SPEC.loader.exec_module(_OPENPI_LOADER)
INFERENCE_IGNORED_AUXILIARY_KIND = _OPENPI_LOADER.INFERENCE_IGNORED_AUXILIARY_KIND
INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA = (
    _OPENPI_LOADER.INFERENCE_VALUE_HEAD_AUXILIARY_SCHEMA
)


def test_verified_target_accepts_only_classified_training_value_head(
    tmp_path,
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
    }
    model = SimpleNamespace(_rlinf_checkpoint_load_report=report)
    assets = SimpleNamespace(weight_paths=(weights,))
    loader_module = SimpleNamespace(
        checkpoint_load_key_classes=_OPENPI_LOADER.checkpoint_load_key_classes
    )
    monkeypatch.setitem(sys.modules, "rlinf.models.embodiment.openpi", loader_module)

    _require_exact_model_load(model, assets)

    report["unexpected_keys"] = (*auxiliary, "other.extra")
    report["unresolved_unexpected_keys"] = ("other.extra",)
    with pytest.raises(ConfigurationError, match="main state_dict"):
        _require_exact_model_load(model, assets)


class _FakeDynamicCache:
    """Small Cache-like type with the HF legacy conversion surface."""

    def __init__(self, legacy) -> None:
        self.legacy = tuple(legacy)

    def to_legacy_cache(self):
        return self.legacy

    @classmethod
    def from_legacy_cache(cls, legacy):
        return cls(legacy)


class _FakeOpenPI(torch.nn.Module):
    """CPU target matching the OpenPI model-space inference surface."""

    def __init__(
        self,
        *,
        horizon: int = 4,
        state_dim: int = 32,
        cache_kind: str = "dynamic",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            action_horizon=horizon,
            action_chunk=horizon,
            action_dim=32,
            action_env_dim=3,
            config_name="pi05_aloha_robotwin",
            pi05=True,
            max_token_len=200,
            num_images_in_input=3,
        )
        self.state_dim = state_dim
        self.cache_kind = cache_kind
        self.encoder_calls = 0
        self.prefill_calls = 0
        self.velocity_calls = 0
        self.velocity_inputs: list[tuple[torch.Tensor, ...]] = []
        language_config = SimpleNamespace(_attn_implementation="sdpa")
        paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=language_config)
        )
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=paligemma,
            forward=self._prefill,
        )

    def _preprocess_observation(self, observation, *, train):
        assert train is False
        batch_size = int(observation.state.shape[0])
        state = torch.zeros(
            (batch_size, self.state_dim),
            device=observation.state.device,
            dtype=observation.state.dtype,
        )
        state[:, : min(self.state_dim, observation.state.shape[1])] = observation.state[
            :, : self.state_dim
        ]
        return [observation.state], [None], None, None, state

    def embed_prefix(self, images, image_masks, language_tokens, language_masks):
        del image_masks, language_tokens, language_masks
        self.encoder_calls += 1
        observation_state = images[0]
        batch_size = int(observation_state.shape[0])
        prefix = torch.zeros(
            (batch_size, 3, 8),
            device=observation_state.device,
            dtype=observation_state.dtype,
        )
        prefix[:, :, 0] = observation_state[:, :1]
        pad = torch.tensor([True, True, False], device=observation_state.device).expand(
            batch_size, 3
        )
        att = torch.tensor(
            [False, False, True], device=observation_state.device
        ).expand(batch_size, 3)
        return prefix, pad, att

    def _prepare_attention_masks_4d(self, masks):
        return masks[:, None]

    def _prefill(self, *, inputs_embeds, **kwargs):
        del kwargs
        self.prefill_calls += 1
        prefix = inputs_embeds[0]
        batch_size = int(prefix.shape[0])
        batch_values = prefix[:, 0, 0].reshape(batch_size, 1, 1, 1)
        key = batch_values.expand(batch_size, 2, 3, 2).clone()
        value = (batch_values + 100).expand(batch_size, 2, 3, 2).clone()
        legacy = ((key, value),)
        if self.cache_kind == "dynamic":
            cache = _FakeDynamicCache(legacy)
        elif self.cache_kind == "tuple":
            cache = legacy
        elif self.cache_kind == "invalid":
            cache = object()
        else:
            raise AssertionError(self.cache_kind)
        return (prefix + 2, None), cache

    def _velocity_formula(self, state, x_t, timestep):
        return 0.25 * x_t + timestep[:, None, None] + 0.01 * state[:, None, :1]

    def get_velocity(
        self,
        state,
        x_t,
        timestep,
        prefix_pad_masks,
        past_key_values,
    ):
        self.velocity_calls += 1
        self.velocity_inputs.append(
            (state, prefix_pad_masks, past_key_values, x_t, timestep)
        )
        return self._velocity_formula(state, x_t, timestep), torch.zeros_like(x_t)


class _FakeDenoiseStepOpenPI(_FakeOpenPI):
    get_velocity = None

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        del prefix_pad_masks, past_key_values
        self.velocity_calls += 1
        return self._velocity_formula(state, x_t, timestep)


def _target(*, family="pi0.5", checkpoint="manifest-sha256:target"):
    return OpenPITargetContract(
        policy_family=family,
        checkpoint_id=checkpoint,
        config_name="pi05_aloha_robotwin" if family == "pi0.5" else "pi0_libero",
        norm_stats_id="physical-intelligence/robotwin@sha256:norm",
        transform_id="LeRobotAlohaDataConfig@12881eda",
        action_horizon=4,
        model_action_dim=32,
        env_action_indices=(2, 0, 7),
        camera_keys=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        model_camera_keys=(
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ),
        max_token_length=200,
    )


def _observation(batch_size=2):
    values = torch.arange(1, batch_size + 1, dtype=torch.float32) * 10
    return SimpleNamespace(state=values[:, None])


def _verifier(model=None, target=None):
    target = target or _target()
    model = model or _FakeOpenPI()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    verified = object.__new__(VerifiedOpenPITarget)
    object.__setattr__(verified, "model", model)
    object.__setattr__(verified, "assets", object())
    object.__setattr__(verified, "contract", target)
    object.__setattr__(verified, "_target_token", object())
    object.__setattr__(
        verified,
        "_verification_proof",
        speculative_module._VERIFIED_OPENPI_TARGET_PROOF,
    )
    return OpenPIParallelVerifier(verified)


def _prediction(verifier, draft, *, target=None):
    return _issue_verified_draft_prediction(
        actions_model=draft,
        target=verifier.verified_target,
        target_contract=target or verifier.target_contract,
        draft_checkpoint_id="derived-draft@revision",
        draft_checkpoint_sha256="ab" * 32,
    )


def _inputs(batch_size=2):
    draft = torch.linspace(-0.4, 0.6, batch_size * 4 * 32).reshape(batch_size, 4, 32)
    noise = torch.linspace(0.7, -0.3, batch_size * 4 * 32).reshape(batch_size, 4, 32)
    times = torch.tensor([0.10, 0.05, 0.025], dtype=torch.float32)
    return draft, noise, times


def test_bk_parallel_matches_per_k_reference_and_preserves_cache_type():
    target = _target()
    model = _FakeOpenPI()
    verifier = _verifier(model, target)
    verifier.begin_episode("episode-parallel")
    bundle = verifier.prepare_prefix(_observation(), episode_id="episode-parallel")
    draft, noise, times = _inputs()

    parallel = verifier.verify(
        bundle,
        _prediction(verifier, draft),
        noise,
        times,
    )

    assert model.encoder_calls == 1
    assert model.prefill_calls == 1
    assert model.velocity_calls == 1
    assert bundle.prefix_embs.shape == (2, 3, 8)
    assert bundle.prefix_pad_masks.shape == (2, 3)
    assert bundle.prefix_att_masks.shape == (2, 3)
    assert bundle.normalized_state.shape == (2, 32)
    assert bundle.contextual_prefix_output.shape == (2, 3, 8)
    assert bundle.kv_cache_type.endswith("._FakeDynamicCache")
    assert parallel.expanded_kv_cache_type == bundle.kv_cache_type
    assert parallel.x0_hat_model.shape == (2, 3, 4, 32)
    assert parallel.x0_hat_env.shape == (2, 3, 4, 3)
    torch.testing.assert_close(
        parallel.x0_hat_env,
        parallel.x0_hat_model[..., [2, 0, 7]],
    )

    state_bk, pad_bk, cache_bk, x_t_bk, timestep_bk = model.velocity_inputs[0]
    assert state_bk.shape == (6, 32)
    assert pad_bk.shape == (6, 3)
    assert type(cache_bk) is _FakeDynamicCache
    assert x_t_bk.shape == (6, 4, 32)
    assert timestep_bk.tolist() == pytest.approx([0.10, 0.05, 0.025, 0.10, 0.05, 0.025])
    assert state_bk[:, 0].tolist() == [10, 10, 10, 20, 20, 20]
    expanded_key = cache_bk.to_legacy_cache()[0][0]
    assert expanded_key[:, 0, 0, 0].tolist() == [10, 10, 10, 20, 20, 20]

    # Independent K=1 calls are the numerical reference.  They reuse one
    # separately prepared prefix and therefore isolate only the batching math.
    reference_model = _FakeOpenPI()
    reference_verifier = _verifier(reference_model, target)
    reference_verifier.begin_episode("episode-reference")
    reference_bundle = reference_verifier.prepare_prefix(
        _observation(), episode_id="episode-reference"
    )
    per_k = []
    for verification_time in times:
        result = reference_verifier.verify(
            reference_bundle,
            _prediction(reference_verifier, draft),
            noise,
            verification_time[None],
        )
        per_k.append(result.x0_hat_model[:, 0])
    sequential = torch.stack(per_k, dim=1)
    assert reference_model.encoder_calls == 1
    assert reference_model.prefill_calls == 1
    assert reference_model.velocity_calls == len(times)
    torch.testing.assert_close(parallel.x0_hat_model, sequential)
    assert parallel.diagnostics["velocity_calls"] == 1
    assert parallel.diagnostics["shared_noise_across_k"] is True
    assert parallel.diagnostics["draft_prediction_asset_verified"] is True


def test_interpolation_reuses_one_noise_per_batch_item_for_every_k():
    target = _target()
    verifier = _verifier(target=target)
    verifier.begin_episode("noise-owner")
    bundle = verifier.prepare_prefix(_observation(), episode_id="noise-owner")
    draft, noise, times = _inputs()

    result = verifier.verify(
        bundle,
        _prediction(verifier, draft),
        noise,
        times,
    )

    expected = (
        times[None, :, None, None] * noise[:, None]
        + (1 - times[None, :, None, None]) * draft[:, None]
    )
    torch.testing.assert_close(result.interpolation_model, expected)
    for batch_index in range(2):
        for k_index in range(3):
            recovered_noise = (
                result.interpolation_model[batch_index, k_index]
                - (1 - times[k_index]) * draft[batch_index]
            ) / times[k_index]
            torch.testing.assert_close(recovered_noise, noise[batch_index])


def test_pi0_draft_target_is_not_blessed_for_pi05_checkpoint():
    target = _target()
    model = _FakeOpenPI()
    verifier = _verifier(model, target)
    verifier.begin_episode("family-gate")
    verifier.prepare_prefix(_observation(), episode_id="family-gate")
    draft, noise, times = _inputs()
    pi0_draft_target = _target(
        family="pi0", checkpoint="hf://openpi/pi0-libero-draft-target@revision"
    )

    with pytest.raises(ConfigurationError, match="policy_family"):
        _prediction(
            verifier,
            draft,
            target=pi0_draft_target,
        )

    assert model.velocity_calls == 0


def test_episode_ownership_reset_and_stale_bundle_contract():
    target = _target()
    verifier = _verifier(target=target)
    other = _verifier(target=target)
    verifier.begin_episode("episode-a")
    bundle = verifier.prepare_prefix(_observation(), episode_id="episode-a")
    draft, noise, times = _inputs()

    with pytest.raises(RuntimeError, match="already active"):
        verifier.begin_episode("episode-b")
    with pytest.raises(RuntimeError, match="another episode"):
        verifier.reset_episode("episode-b")

    other.begin_episode("episode-a")
    with pytest.raises(RuntimeError, match="different verifier"):
        other.verify(
            bundle,
            _prediction(other, draft),
            noise,
            times,
        )

    verifier.reset_episode("episode-a")
    with pytest.raises(RuntimeError, match="stale|invalidated"):
        verifier.verify(
            bundle,
            _prediction(verifier, draft),
            noise,
            times,
        )
    verifier.begin_episode("episode-b")
    with pytest.raises(RuntimeError, match="does not own"):
        verifier.prepare_prefix(_observation(), episode_id="episode-a")


def test_second_prefix_invalidates_the_old_same_episode_bundle():
    target = _target()
    verifier = _verifier(target=target)
    verifier.begin_episode("moving-camera-frame")
    old = verifier.prepare_prefix(_observation(), episode_id="moving-camera-frame")
    current = verifier.prepare_prefix(_observation(), episode_id="moving-camera-frame")
    draft, noise, times = _inputs()

    with pytest.raises(RuntimeError, match="stale|invalidated"):
        verifier.verify(
            old,
            _prediction(verifier, draft),
            noise,
            times,
        )
    verifier.verify(
        current,
        _prediction(verifier, draft),
        noise,
        times,
    )


@pytest.mark.parametrize(
    ("mutator", "error", "message"),
    [
        (
            lambda draft, noise, times: (draft[:, :, :-1], noise, times),
            ShapeMismatchError,
            "draft_model",
        ),
        (
            lambda draft, noise, times: (
                draft,
                noise.to(dtype=torch.float64),
                times,
            ),
            TypeError,
            "exact dtype",
        ),
        (
            lambda draft, noise, times: (
                draft,
                noise,
                times.to(dtype=torch.float64),
            ),
            TypeError,
            "exact dtype",
        ),
        (
            lambda draft, noise, times: (
                draft.clone().index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor(float("nan")),
                ),
                noise,
                times,
            ),
            ValueError,
            "NaN",
        ),
        (
            lambda draft, noise, times: (
                draft,
                noise,
                torch.tensor([0.0], dtype=torch.float32),
            ),
            ValueError,
            "\\(0, 1\\]",
        ),
    ],
)
def test_verification_inputs_fail_closed(mutator, error, message):
    target = _target()
    verifier = _verifier(target=target)
    verifier.begin_episode("bad-input")
    bundle = verifier.prepare_prefix(_observation(), episode_id="bad-input")
    args = mutator(*_inputs())

    with pytest.raises(error, match=message):
        draft, noise, times = args
        verifier.verify(
            bundle,
            _prediction(verifier, draft),
            noise,
            times,
        )


def test_prefix_state_and_cache_fail_closed_before_verification():
    target = _target()
    wrong_state = _verifier(_FakeOpenPI(state_dim=31), target)
    wrong_state.begin_episode("bad-state")
    with pytest.raises(ShapeMismatchError, match="normalized_state"):
        wrong_state.prepare_prefix(_observation(), episode_id="bad-state")

    invalid_cache = _verifier(_FakeOpenPI(cache_kind="invalid"), target)
    invalid_cache.begin_episode("bad-cache")
    with pytest.raises(TypeError, match="past_key_values"):
        invalid_cache.prepare_prefix(_observation(), episode_id="bad-cache")


def test_legacy_tuple_cache_and_denoise_step_path_are_preserved():
    target = _target()
    model = _FakeDenoiseStepOpenPI(cache_kind="tuple")
    verifier = _verifier(model, target)
    verifier.begin_episode("legacy-cache")
    bundle = verifier.prepare_prefix(_observation(), episode_id="legacy-cache")
    draft, noise, times = _inputs()
    result = verifier.verify(
        bundle,
        _prediction(verifier, draft),
        noise,
        times,
    )

    assert bundle.kv_cache_type == "builtins.tuple"
    assert result.expanded_kv_cache_type == "builtins.tuple"
    assert result.velocity_path == "model.denoise_step"
    assert model.velocity_calls == 1


def test_checkpoint_and_model_geometry_are_bound_at_construction():
    target = _target()
    with pytest.raises(TypeError, match="VerifiedOpenPITarget"):
        OpenPIParallelVerifier(_FakeOpenPI())
    with pytest.raises(ConfigurationError, match="action_horizon"):
        _verifier(_FakeOpenPI(horizon=5), target)
    with pytest.raises(ConfigurationError, match="config_name"):
        _verifier(_FakeOpenPI(), replace(target, config_name="pi05_libero"))


def test_existing_torch_postprocess_consumes_explicit_environment_dims():
    target = _target()
    verifier = _verifier(target=target)
    verifier.begin_episode("postprocess")
    bundle = verifier.prepare_prefix(_observation(), episode_id="postprocess")
    draft, noise, times = _inputs()
    verification = verifier.verify(
        bundle,
        _prediction(verifier, draft),
        noise,
        times,
    )
    config = TritonSpeculativeConfig(
        absolute_error_threshold=0.2,
        relative_error_threshold=0.5,
        distance_indices=(0, 1, 2),
    )

    results = triton_postprocess_batch(verification, config, backend="torch")

    assert len(results) == 2
    assert all(result.backend == "torch" for result in results)
    assert all(result.x0_hat.shape == (3, 4, 3) for result in results)
    torch.testing.assert_close(results[0].x0_hat, verification.x0_hat_env[0])


def test_contract_requires_explicit_32d_model_and_environment_selection():
    with pytest.raises(ValueError, match="32D"):
        replace(_target(), state_dim=14)
    with pytest.raises(ValueError, match="explicitly select"):
        replace(_target(), env_action_indices=())
    with pytest.raises(ValueError, match="unique"):
        replace(_target(), env_action_indices=(0, 0))
