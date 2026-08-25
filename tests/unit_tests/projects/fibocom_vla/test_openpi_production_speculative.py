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

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlinf.projects.fibocom_vla.config import SpeculativeConfig  # noqa: E402
from rlinf.projects.fibocom_vla.contracts import (  # noqa: E402
    ActionChunk,
    Observation,
    PolicyOutput,
    RobotState,
)
from rlinf.projects.fibocom_vla.errors import ConfigurationError  # noqa: E402
from rlinf.projects.fibocom_vla.inference import (  # noqa: E402
    openpi_production_speculative as production_module,
)
from rlinf.projects.fibocom_vla.inference import (  # noqa: E402
    openpi_speculative as speculative_module,
)
from rlinf.projects.fibocom_vla.inference.openpi_production_speculative import (  # noqa: E402
    OpenPIProductionSpeculativePolicy,
)
from rlinf.projects.fibocom_vla.inference.openpi_speculative import (  # noqa: E402
    OpenPIParallelVerification,
    OpenPITargetContract,
    VerifiedDraftPrediction,
    VerifiedOpenPITarget,
)
from rlinf.projects.fibocom_vla.inference.pi05_draft import (  # noqa: E402
    Pi05DraftProductionAdapter,
)
from rlinf.projects.fibocom_vla.openpi_adapter import (  # noqa: E402
    OpenPiAdapterConfig,
    RLinfOpenPiChunkPolicy,
)

HORIZON = 50
MODEL_DIM = 32
ENV_DIM = 14
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SEMANTICS = tuple(f"joint_{index}" for index in range(ENV_DIM))


class _FakeProductionOpenPI(torch.nn.Module):
    """Transform-complete fake; verification itself is supplied separately."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(
            config_name="pi05_aloha_robotwin",
            action_horizon=HORIZON,
            action_chunk=HORIZON,
            action_dim=MODEL_DIM,
            action_env_dim=ENV_DIM,
            max_token_len=200,
            num_images_in_input=3,
            pi05=True,
        )
        self.transform_calls: list[str] = []
        self.output_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.noise_objects: list[torch.Tensor] = []
        self.eval()

    def obs_processor(self, observation):
        self.transform_calls.append("obs_processor")
        assert observation["main_images"].shape == (1, 6, 8, 3)
        assert observation["wrist_images"].shape == (1, 2, 6, 8, 3)
        assert observation["states"].shape == (1, ENV_DIM)
        return {
            "state": observation["states"],
            "images": observation["main_images"],
            "prompt": observation["task_descriptions"],
        }

    def input_transform(self, observation, *, transpose):
        self.transform_calls.append("input_transform")
        assert transpose is False
        state = torch.zeros((1, MODEL_DIM), dtype=torch.float32)
        state[:, :ENV_DIM] = observation["state"].to(dtype=torch.float32)
        return {"state": state, "images": observation["images"], "stage": "input"}

    def precision_processor(self, observation):
        self.transform_calls.append("precision_processor")
        assert observation["stage"] == "input"
        return {**observation, "stage": "precision"}

    def observation_from_dict(self, observation):
        self.transform_calls.append("observation_from_dict")
        assert observation["stage"] == "precision"
        return SimpleNamespace(state=observation["state"], payload=observation)

    def sample_noise(self, shape, device):
        noise = torch.zeros(shape, device=device, dtype=torch.float32)
        self.noise_objects.append(noise)
        return noise

    def output_transform(self, outputs):
        actions = outputs["actions"]
        state = outputs["state"]
        assert tuple(actions.shape) == (1, HORIZON, MODEL_DIM)
        assert tuple(state.shape) == (1, MODEL_DIM)
        self.output_inputs.append((actions.detach().clone(), state.detach().clone()))
        # State dependence catches attempts to transform before the complete
        # normalized action and then splice in environment space.
        transformed = actions[:, :, :ENV_DIM] * 3.0 + state[:, None, :ENV_DIM]
        return {"actions": transformed}


class _FakeDraftTargetContract:
    def __init__(self, target: VerifiedOpenPITarget) -> None:
        self.target = target
        self.seen: list[VerifiedOpenPITarget] = []

    def require_matches_verified_openpi(self, target: VerifiedOpenPITarget) -> None:
        self.seen.append(target)
        if target is not self.target:
            raise ConfigurationError("wrong exact target")


class _FakeProductionDraftHead(torch.nn.Module):
    def __init__(self, target: VerifiedOpenPITarget, actions: torch.Tensor) -> None:
        super().__init__()
        self.production_compatible = True
        self.target_contract = _FakeDraftTargetContract(target)
        self._production_artifact_manifest_sha256 = "ab" * 32
        self._production_weights_sha256 = "cd" * 32
        self.actions = actions
        self.inputs: list[dict[str, torch.Tensor]] = []
        self.eval()
        self.requires_grad_(False)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        assert tuple(inputs["prefix_embs"].shape) == (1, 4, 2048)
        assert tuple(inputs["prefix_pad_masks"].shape) == (1, 4)
        assert tuple(inputs["prefix_att_masks"].shape) == (1, 4)
        assert tuple(inputs["robot_state"].shape) == (1, MODEL_DIM)
        assert tuple(inputs["last_actions"].shape) == (1, HORIZON, MODEL_DIM)
        self.inputs.append(
            {name: value.detach().clone() for name, value in inputs.items()}
        )
        return self.actions.to(
            device=inputs["robot_state"].device,
            dtype=inputs["robot_state"].dtype,
        ).clone()


class _FakeParallelVerifier:
    def __init__(
        self,
        target: VerifiedOpenPITarget,
        main_candidate: torch.Tensor,
    ) -> None:
        self.verified_target = target
        self.model = target.model
        self.target_contract = target.contract
        self._velocity_target = None
        self.main_candidate = main_candidate
        self.active_episode: str | None = None
        self.begin_calls: list[str] = []
        self.reset_calls: list[str | None] = []
        self.model_observations: list[object] = []
        self.predictions: list[VerifiedDraftPrediction] = []
        self.noise_ids: list[int] = []

    def begin_episode(self, episode_id: str) -> None:
        if self.active_episode is not None:
            raise RuntimeError("episode already active")
        self.active_episode = episode_id
        self.begin_calls.append(episode_id)

    def reset_episode(self, episode_id: str | None = None) -> None:
        if episode_id is not None and episode_id != self.active_episode:
            raise RuntimeError("wrong episode")
        self.reset_calls.append(episode_id)
        self.active_episode = None

    def prepare_prefix(self, observation, *, episode_id: str):
        assert episode_id == self.active_episode
        self.model_observations.append(observation)
        state = observation.state
        return SimpleNamespace(
            prefix_embs=torch.full((1, 4, 2048), 2.0, dtype=state.dtype),
            prefix_pad_masks=torch.tensor([[True, True, True, False]]),
            prefix_att_masks=torch.tensor([[False, False, False, True]]),
            normalized_state=state,
            batch_size=1,
            device=state.device,
        )

    def verify(self, bundle, prediction, noise, verification_times):
        assert isinstance(prediction, VerifiedDraftPrediction)
        assert prediction._target_token is self.verified_target._target_token
        assert tuple(noise.shape) == (1, HORIZON, MODEL_DIM)
        assert noise is self.model.noise_objects[-1]
        assert bundle.normalized_state.device == noise.device
        self.predictions.append(prediction)
        self.noise_ids.append(id(noise))

        draft = prediction.actions_model
        count = int(verification_times.numel())
        candidate = self.main_candidate.to(device=draft.device, dtype=draft.dtype)
        candidate = candidate[None, None].expand(1, count, -1, -1).clone()
        interpolation = (
            verification_times.view(1, count, 1, 1) * noise[:, None]
            + (1 - verification_times.view(1, count, 1, 1)) * draft[:, None]
        )
        velocity = (interpolation - candidate) / verification_times.view(1, count, 1, 1)
        env = torch.tensor(
            self.target_contract.env_action_indices,
            dtype=torch.long,
            device=draft.device,
        )
        return OpenPIParallelVerification(
            draft_model=draft,
            shared_noise=noise,
            verification_times=verification_times,
            interpolation_model=interpolation,
            predicted_velocity_model=velocity,
            x0_hat_model=candidate,
            draft_env=draft.index_select(-1, env),
            shared_noise_env=noise.index_select(-1, env),
            predicted_velocity_env=velocity.index_select(-1, env),
            x0_hat_env=candidate.index_select(-1, env),
            env_action_indices=self.target_contract.env_action_indices,
            velocity_path="model.get_velocity",
            kv_cache_type="fake.Cache",
            expanded_kv_cache_type="fake.Cache",
            episode_id=self.active_episode,
            target_contract=self.target_contract,
            draft_checkpoint_id=prediction.draft_checkpoint_id,
            draft_checkpoint_sha256=prediction.draft_checkpoint_sha256,
        )


def _contract() -> OpenPITargetContract:
    return OpenPITargetContract(
        policy_family="pi0.5",
        checkpoint_id="pi05@weights-sha256:" + "12" * 32,
        config_name="pi05_aloha_robotwin",
        norm_stats_id="physical-intelligence/robotwin@sha256:" + "34" * 32,
        transform_id="sha256:" + "56" * 32,
        action_horizon=HORIZON,
        model_action_dim=MODEL_DIM,
        env_action_indices=tuple(range(ENV_DIM)),
        camera_keys=CAMERAS,
        model_camera_keys=(
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ),
        environment_action_semantics=SEMANTICS,
        delta_action_mask=(True,) * 6 + (False,) + (True,) * 6 + (False,),
        max_token_length=200,
        normalization="quantile_q01_q99",
        asset_id="physical-intelligence/robotwin",
    )


def _target(model: _FakeProductionOpenPI | None = None) -> VerifiedOpenPITarget:
    model = model or _FakeProductionOpenPI()
    model.eval()
    model.requires_grad_(False)
    target = object.__new__(VerifiedOpenPITarget)
    object.__setattr__(target, "model", model)
    object.__setattr__(target, "assets", object())
    object.__setattr__(target, "contract", _contract())
    object.__setattr__(target, "_target_token", object())
    object.__setattr__(
        target,
        "_verification_proof",
        speculative_module._VERIFIED_OPENPI_TARGET_PROOF,
    )
    return target


def _draft_adapter(
    target: VerifiedOpenPITarget,
    actions: torch.Tensor,
) -> tuple[Pi05DraftProductionAdapter, _FakeProductionDraftHead]:
    head = _FakeProductionDraftHead(target, actions)
    adapter = object.__new__(Pi05DraftProductionAdapter)
    adapter._head = head
    adapter._owner_token = object()
    return adapter, head


def _main_policy(target: VerifiedOpenPITarget) -> RLinfOpenPiChunkPolicy:
    return RLinfOpenPiChunkPolicy(
        target.model,
        OpenPiAdapterConfig(
            main_camera=CAMERAS[0],
            wrist_cameras=CAMERAS[1:],
            period_s=0.02,
            expected_state_names=SEMANTICS,
        ),
    )


def _observation() -> Observation:
    images = {
        CAMERAS[0]: np.full((6, 8, 3), 1, dtype=np.uint8),
        CAMERAS[1]: np.full((6, 8, 3), 2, dtype=np.uint8),
        CAMERAS[2]: np.full((6, 8, 3), 3, dtype=np.uint8),
    }
    return Observation(
        state=RobotState(
            np.linspace(0.1, 1.4, ENV_DIM, dtype=np.float32),
            SEMANTICS,
        ),
        images=images,
        instruction="stack the red block",
    )


def _install_verifier(monkeypatch, target, main_candidate):
    constructed = []

    def factory(*args, **kwargs):
        # The production policy must not pass velocity_target or any other
        # arbitrary callable into OpenPIParallelVerifier.
        assert args == (target,)
        assert kwargs == {}
        verifier = _FakeParallelVerifier(target, main_candidate)
        constructed.append(verifier)
        return verifier

    monkeypatch.setattr(production_module, "OpenPIParallelVerifier", factory)
    return constructed


def _config(**overrides) -> SpeculativeConfig:
    values = {
        "absolute_error_threshold": 0.0,
        "relative_error_threshold": 0.0,
        "minimum_prefix": 2,
        "fallback_acceptance_ratio": 0.25,
        "fallback_cooldown_chunks": 0,
    }
    values.update(overrides)
    return SpeculativeConfig(**values)


def test_verified_path_uses_signed_draft_real_transforms_and_main_only_grippers(
    monkeypatch,
):
    target = _target()
    draft_actions = torch.zeros((1, HORIZON, MODEL_DIM), dtype=torch.float32)
    main_candidate = torch.zeros((HORIZON, MODEL_DIM), dtype=torch.float32)
    main_candidate[:, 6] = 4.0
    main_candidate[:, 13] = -3.0
    constructed = _install_verifier(monkeypatch, target, main_candidate)
    draft, head = _draft_adapter(target, draft_actions)
    policy = OpenPIProductionSpeculativePolicy(
        target,
        draft,
        _main_policy(target),
        _config(),
        episode_id="episode-1",
        verification_times=(0.5, 0.25),
        backend="torch",
    )

    output = policy.predict(_observation())

    verifier = constructed[0]
    assert verifier.begin_calls == ["episode-1"]
    assert len(verifier.predictions) == 1
    assert verifier.predictions[0]._target_token is target._target_token
    assert head.target_contract.seen == [target, target, target]
    assert torch.count_nonzero(head.inputs[0]["last_actions"]) == 0
    assert target.model.transform_calls == [
        "obs_processor",
        "input_transform",
        "precision_processor",
        "observation_from_dict",
    ]
    assert len(target.model.output_inputs) == 1
    full_model_input, transformed_state = target.model.output_inputs[0]
    assert tuple(full_model_input.shape) == (1, HORIZON, MODEL_DIM)
    torch.testing.assert_close(full_model_input[0, :, 6], torch.full((HORIZON,), 4.0))
    torch.testing.assert_close(full_model_input[0, :, 13], torch.full((HORIZON,), -3.0))
    assert output.accepted_prefix == HORIZON
    assert output.path == "openpi_production_speculative_verified"
    assert output.diagnostics["distance_indices"] == tuple(
        index for index in range(ENV_DIM) if index not in (6, 13)
    )
    assert output.diagnostics["gripper_source"] == "main_candidate_only"
    assert output.diagnostics["rtc_native_supported"] is False
    assert policy.rtc_native_supported is False
    np.testing.assert_allclose(output.action.model_values[:, 6], 4.0)
    np.testing.assert_allclose(output.action.model_values[:, 13], -3.0)
    state = transformed_state[0, :ENV_DIM].numpy()
    np.testing.assert_allclose(output.action.values[:, 6], 12.0 + state[6])
    np.testing.assert_allclose(output.action.values[:, 13], -9.0 + state[13])

    # The previous complete 32-D chunk is the next signed Draft call's history.
    second = policy.predict(_observation())
    assert second.accepted_prefix == HORIZON
    torch.testing.assert_close(
        head.inputs[1]["last_actions"][0],
        torch.from_numpy(output.action.model_values),
    )
    assert len(set(verifier.noise_ids)) == 2


def test_low_acceptance_calls_same_round_full_main_and_commits_its_model_chunk(
    monkeypatch,
):
    target = _target()
    draft_actions = torch.zeros((1, HORIZON, MODEL_DIM), dtype=torch.float32)
    main_candidate = torch.ones((HORIZON, MODEL_DIM), dtype=torch.float32)
    _install_verifier(monkeypatch, target, main_candidate)
    draft, _ = _draft_adapter(target, draft_actions)
    full_main_calls = []
    fallback_env = np.full((HORIZON, ENV_DIM), 7.0, dtype=np.float32)
    fallback_model = np.full((HORIZON, MODEL_DIM), 8.0, dtype=np.float32)

    def full_main(self, observation):
        full_main_calls.append((self, observation))
        return PolicyOutput(
            action=ActionChunk(
                values=fallback_env,
                period_s=self.config.period_s,
                source_observation_ns=observation.timestamp_ns,
                model_values=fallback_model,
            ),
            model_latency_ms=76.0,
            path="rlinf_openpi_native",
            diagnostics={"full_main_fake": True},
        )

    monkeypatch.setattr(RLinfOpenPiChunkPolicy, "predict", full_main)
    policy = OpenPIProductionSpeculativePolicy(
        target,
        draft,
        _main_policy(target),
        _config(),
        episode_id="episode-fallback",
        verification_times=(0.5,),
        backend="torch",
    )

    output = policy.predict(_observation())

    assert len(full_main_calls) == 1
    assert output.accepted_prefix == 0
    assert output.path == "openpi_production_speculative_fallback_full_main"
    assert output.diagnostics["fallback_semantics"] == (
        "same_round_full_main_denoising"
    )
    assert output.diagnostics["used_full_fallback"] is True
    np.testing.assert_array_equal(output.action.values, fallback_env)
    np.testing.assert_array_equal(output.action.model_values, fallback_model)
    torch.testing.assert_close(
        policy.previous_model_actions,
        torch.from_numpy(fallback_model),
    )
    assert target.model.output_inputs == []


def test_target_model_identity_and_frozen_state_fail_closed(monkeypatch):
    target = _target()
    draft_actions = torch.zeros((1, HORIZON, MODEL_DIM), dtype=torch.float32)
    main_candidate = torch.zeros((HORIZON, MODEL_DIM), dtype=torch.float32)
    _install_verifier(monkeypatch, target, main_candidate)
    draft, _ = _draft_adapter(target, draft_actions)
    wrong_main = _main_policy(_target())

    with pytest.raises(ConfigurationError, match="identity"):
        OpenPIProductionSpeculativePolicy(
            target,
            draft,
            wrong_main,
            _config(),
            episode_id="wrong-main",
        )

    main = _main_policy(target)
    policy = OpenPIProductionSpeculativePolicy(
        target,
        draft,
        main,
        _config(),
        episode_id="frozen",
        backend="torch",
    )
    target.model.train()
    with pytest.raises(ConfigurationError, match="eval-mode and frozen"):
        policy.predict(_observation())


def test_reset_clears_previous_actions_and_predict_reset_are_non_reentrant(
    monkeypatch,
):
    target = _target()
    draft_actions = torch.zeros((1, HORIZON, MODEL_DIM), dtype=torch.float32)
    main_candidate = torch.zeros((HORIZON, MODEL_DIM), dtype=torch.float32)
    constructed = _install_verifier(monkeypatch, target, main_candidate)
    draft, _ = _draft_adapter(target, draft_actions)
    policy = OpenPIProductionSpeculativePolicy(
        target,
        draft,
        _main_policy(target),
        _config(),
        episode_id="before-reset",
        backend="torch",
    )
    policy.predict(_observation())
    assert policy.previous_model_actions is not None

    policy.reset_episode("after-reset")

    verifier = constructed[0]
    assert verifier.reset_calls == ["before-reset"]
    assert verifier.begin_calls == ["before-reset", "after-reset"]
    assert policy.episode_id == "after-reset"
    assert policy.previous_model_actions is None

    assert policy._predict_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="not reentrant"):
            policy.predict(_observation())
        with pytest.raises(RuntimeError, match="not reentrant"):
            policy.reset_episode("blocked-reset")
    finally:
        policy._predict_lock.release()
