import math

import pytest
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_vla_corrector import VLACorrectorConfig, VLACorrectorPolicy
from lerobot_policy_vla_corrector.corrector import MonitorStep
from lerobot_policy_vla_corrector.modeling_vla_corrector import TELEMETRY_KEYS
from tests.stub import ACTION_DIM, CHUNK, IMG, N_ACTION_STEPS, TOKEN_DIM, StubBackend, make_stub, observation, stub_features


def make_policy(**overrides) -> VLACorrectorPolicy:
    inputs, outputs = stub_features()
    kwargs = dict(
        input_features=inputs,
        output_features=outputs,
        device="cpu",
        mlp_widths=(16, 16),
        action_embed_dim=4,
        k_step=2,
        bootstrap_size=3,
        window=3,
        noise_floor=10.0,
        jump_trigger=math.inf,
    )
    kwargs.update(overrides)
    cfg = VLACorrectorConfig(**kwargs)
    torch.manual_seed(0)
    base = make_stub()
    return VLACorrectorPolicy(cfg, base=base, backend=StubBackend(base))


def noise_for(step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randn(1, CHUNK, ACTION_DIM, generator=g)


def rollout(policy, steps: int, seed: int = 0) -> torch.Tensor:
    policy.reset()
    out = []
    for t in range(steps):
        out.append(policy.select_action(observation(t, seed=seed), noise=noise_for(t)))
    return torch.stack(out)


def test_registered_through_plugin_discovery():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    assert "vla_corrector" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("vla_corrector") is VLACorrectorConfig
    assert VLACorrectorPolicy.name == "vla_corrector"


def test_config_rejects_bad_values():
    inputs, outputs = stub_features()
    with pytest.raises(ValueError):
        VLACorrectorConfig(input_features=inputs, output_features=outputs, device="cpu", mode="guided")
    with pytest.raises(ValueError):
        VLACorrectorConfig(input_features=inputs, output_features=outputs, device="cpu", k_step=5, monitor_every_n_steps=2)
    with pytest.raises(ValueError):
        VLACorrectorConfig(input_features=inputs, output_features=outputs, device="cpu", lambda_on=2.0, lambda_off=3.0)


def test_delta_indices():
    policy = make_policy(k_step=4)
    cfg = policy.config
    assert cfg.image_observation_delta_indices == [0, 4]
    assert cfg.state_observation_delta_indices == [0]
    assert cfg.action_delta_indices == [0]
    assert cfg.reward_delta_indices is None
    assert cfg.token_dim == TOKEN_DIM


def test_mode_off_is_bit_identical_to_the_base_over_200_steps():
    policy = make_policy(mode="off")
    base = make_stub()
    base.reset()
    ours = rollout(policy, 200)
    ref = torch.stack([base.select_action(observation(t), noise=noise_for(t)) for t in range(200)])
    assert torch.equal(ours, ref)
    assert float((ours - ref).abs().max()) == 0.0


def test_monitor_only_keeps_the_base_actions_and_logs_telemetry():
    policy = make_policy(mode="monitor_only")
    off = make_policy(mode="off")
    a = rollout(policy, 40)
    b = rollout(off, 40)
    assert torch.equal(a, b)
    assert len(policy.telemetry) == 40
    assert set(TELEMETRY_KEYS) <= set(policy.telemetry[-1])
    scores = [row["score"] for row in policy.telemetry]
    assert all(math.isnan(s) for s in scores[:2])
    assert all(math.isfinite(s) for s in scores[2:])
    assert not any(row["truncated"] for row in policy.telemetry)


def force_trigger(policy, at_step: int):
    original = policy.monitor.observe

    def observe(step, z_t):
        res = original(step, z_t)
        if step == at_step:
            dp = torch.ones_like(z_t)
            dr = -torch.ones_like(z_t)
            return MonitorStep(step, 2.0, res.threshold, res.release, True, dr, dp)
        return res

    policy.monitor.observe = observe


def test_truncate_clears_the_queue_and_replans_at_the_trigger():
    policy = make_policy(mode="truncate")
    force_trigger(policy, 4)
    rollout(policy, 12)
    rows = policy.telemetry
    assert rows[4]["triggered"] and rows[4]["replanned"]
    assert rows[4]["truncated"] == N_ACTION_STEPS - 4
    assert [r["replanned"] for r in rows[:12]] == [True, False, False, False, True] + [False] * 7
    assert rows[4]["ogg_calls"] == 0


def test_truncate_ogg_runs_one_guided_replan_after_the_trigger():
    policy = make_policy(mode="truncate_ogg", ogg_eta=1.0, ogg_calls_after_trigger=1)
    plain = make_policy(mode="truncate")
    force_trigger(policy, 4)
    force_trigger(plain, 4)
    a = rollout(policy, 30)
    b = rollout(plain, 30)
    rows = policy.telemetry
    assert rows[4]["ogg_calls"] == 1 and rows[29]["ogg_calls"] == 1
    assert math.isfinite(rows[4]["ogg_loss"]) and math.isfinite(rows[4]["ogg_grad_norm"])
    assert torch.equal(a[:4], b[:4])
    assert not torch.equal(a[4:14], b[4:14])
    assert torch.equal(a[14:], b[14:])


def test_truncate_ogg_with_eta_zero_matches_truncate():
    policy = make_policy(mode="truncate_ogg", ogg_eta=0.0)
    plain = make_policy(mode="truncate")
    force_trigger(policy, 4)
    force_trigger(plain, 4)
    assert torch.equal(rollout(policy, 30), rollout(plain, 30))


def test_first_predict_action_chunk_after_reset():
    policy = make_policy(mode="truncate_ogg")
    policy.reset()
    chunk = policy.predict_action_chunk(observation(0), noise=noise_for(0))
    assert chunk.shape == (1, CHUNK, ACTION_DIM)
    policy.reset()
    again = policy.predict_action_chunk(observation(0), noise=noise_for(0))
    assert torch.equal(chunk, again)


def test_reset_clears_state():
    policy = make_policy(mode="monitor_only")
    rollout(policy, 7)
    policy.reset()
    assert policy._step == 0 and len(policy._queue) == 0 and len(policy.monitor.queue) == 0


def test_save_and_load_keep_only_the_corrector(tmp_path):
    policy = make_policy(mode="truncate_ogg")
    with torch.no_grad():
        for p in policy.corrector.parameters():
            p.add_(0.01)
    policy.save_pretrained(tmp_path)
    from safetensors.torch import load_file

    state = load_file(str(tmp_path / "model.safetensors"))
    assert all(k.startswith("corrector.") for k in state)
    assert "corrector.latent_mean" in state and "corrector.latent_std" in state
    base = make_stub()
    loaded = VLACorrectorPolicy.from_pretrained(tmp_path, base=base, backend=StubBackend(base))
    for (k1, v1), (k2, v2) in zip(
        policy.corrector.state_dict().items(), loaded.corrector.state_dict().items(), strict=True
    ):
        assert k1 == k2 and torch.equal(v1, v2)
    assert loaded.config.mode == "truncate_ogg"
    force_trigger(policy, 4)
    force_trigger(loaded, 4)
    assert torch.equal(rollout(policy, 20), rollout(loaded, 20))


def test_forward_trains_only_the_corrector():
    policy = make_policy()
    policy.train()
    b = 3
    batch = {
        IMG: torch.rand(b, 2, 3, 4, 4),
        f"{IMG}_is_pad": torch.tensor([[False, False], [False, True], [False, False]]),
        OBS_STATE: torch.randn(b, 4),
        ACTION: torch.randn(b, 1, ACTION_DIM),
    }
    loss, info = policy.forward(batch)
    assert torch.isfinite(loss)
    assert info["n_valid"] == 2
    loss.backward()
    assert all(p.grad is not None for p in policy.corrector.parameters())
    assert all(p.grad is None for p in policy.base.parameters())
    assert [id(p) for p in policy.get_optim_params()] == [id(p) for p in policy.corrector.parameters()]


def test_monitor_every_two_steps():
    policy = make_policy(mode="monitor_only", k_step=2, monitor_every_n_steps=2)
    rollout(policy, 10)
    scores = [row["score"] for row in policy.telemetry]
    assert all(math.isnan(scores[t]) for t in range(1, 10, 2))
    assert all(math.isfinite(scores[t]) for t in range(2, 10, 2))


def test_batched_inference_is_refused_unless_off():
    policy = make_policy(mode="monitor_only")
    with pytest.raises(ValueError):
        policy.select_action(observation(0, b=2))
    policy = make_policy(mode="off")
    assert policy.select_action(observation(0, b=2)).shape == (2, ACTION_DIM)
