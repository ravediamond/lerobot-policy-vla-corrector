import math

import torch

from lerobot_policy_vla_corrector.corrector import (
    LatentDynamicsMLP,
    Monitor,
    OGGProcessor,
    ThresholdConfig,
    ThresholdState,
    dynamics_loss,
    guided_euler,
    mismatch_score,
)


def run(state: ThresholdState, scores):
    out = []
    for s in scores:
        out.append(state.step(s))
    return out


def test_bootstrap_never_triggers_before_the_window_is_full():
    cfg = ThresholdConfig(bootstrap_size=15)
    state = ThresholdState(cfg)
    results = run(state, [5.0] * 15)
    assert all(not trig for _, trig in results)
    assert results[0][0] == cfg.bootstrap_threshold
    assert state.bootstrapped


def clean_state(cfg: ThresholdConfig, level: float = 0.3) -> ThresholdState:
    state = ThresholdState(cfg)
    run(state, [level] * (cfg.bootstrap_size + cfg.window))
    return state


def test_patience_counts_consecutive_steps_over_the_threshold():
    cfg = ThresholdConfig(patience=5, jump_trigger=math.inf, cooldown=10, ewma_alpha=1.0)
    state = clean_state(cfg)
    high = state.threshold + 1.0
    results = run(state, [high] * 4)
    assert all(not trig for _, trig in results)
    _, trig = state.step(high)
    assert trig
    state = clean_state(cfg)
    results = run(state, [high, high, high, 0.3, high, high, high, high])
    assert all(not trig for _, trig in results)


def test_jump_triggers_at_once():
    cfg = ThresholdConfig(patience=5, jump_trigger=0.06)
    state = clean_state(cfg)
    _, trig = state.step(state.threshold + 1.0)
    assert trig


def test_cooldown_blocks_retrigger_except_hard():
    cfg = ThresholdConfig(patience=1, jump_trigger=math.inf, cooldown=10, hard_retrigger_steps=3, ewma_alpha=1.0)
    state = clean_state(cfg)
    mild = state.threshold + cfg.trigger_margin + 0.01
    _, trig = state.step(mild)
    assert trig
    results = run(state, [mild] * 10)
    assert all(not trig for _, trig in results)
    _, trig = state.step(mild)
    assert trig
    state = clean_state(cfg)
    state.step(mild)
    hard = state.threshold + cfg.hard_retrigger_margin + 0.01
    results = run(state, [hard, hard, hard])
    assert [trig for _, trig in results] == [False, False, True]


def test_hysteresis_release_below_lambda_off():
    cfg = ThresholdConfig(patience=1, jump_trigger=math.inf, cooldown=0, reset_steps=5, ewma_alpha=1.0)
    state = clean_state(cfg)
    assert state.release <= state.threshold
    state.step(state.threshold + 1.0)
    assert state.alert
    run(state, [0.2] * 4)
    assert state.alert
    state.step(0.2)
    assert not state.alert


def test_mismatch_score_bounds():
    a = torch.randn(4, 6)
    assert abs(mismatch_score(a, a)) < 1e-6
    assert abs(mismatch_score(a, -a) - 2.0) < 1e-6
    assert mismatch_score(torch.zeros(3), a[0, :3]) == 1.0


def test_monitor_aligns_predictions_k_steps_ahead():
    k = 3
    dyn = lambda z, a: torch.zeros_like(z) + a.sum()  # noqa: E731
    mon = Monitor(dyn, k_step=k, thresholds=ThresholdConfig(), every_n=1)
    zs = [torch.randn(2, 4) for _ in range(10)]
    scores = []
    for t in range(10):
        res = mon.observe(t, zs[t])
        scores.append(res.score)
        mon.predict(t, zs[t], torch.ones(2))
    assert all(math.isnan(s) for s in scores[:k])
    assert all(math.isfinite(s) for s in scores[k:])
    assert len(mon.queue) == k


def test_monitor_every_n_keeps_alignment():
    k = 4
    dyn = lambda z, a: torch.ones_like(z)  # noqa: E731
    mon = Monitor(dyn, k_step=k, thresholds=ThresholdConfig(), every_n=2)
    seen = []
    for t in range(12):
        if mon.active(t):
            res = mon.observe(t, torch.randn(2, 3))
            seen.append((t, math.isfinite(res.score)))
            mon.predict(t, torch.randn(2, 3), torch.ones(2))
    assert [t for t, _ in seen] == [0, 2, 4, 6, 8, 10]
    assert [ok for _, ok in seen] == [False, False, True, True, True, True]


def test_mlp_shapes_and_parameter_count():
    mlp = LatentDynamicsMLP(token_dim=8, action_dim=3, widths=(16, 16), action_embed_dim=4)
    z = torch.randn(5, 8)
    a = torch.randn(3)
    assert mlp(z, a).shape == (5, 8)
    assert mlp(z.unsqueeze(0), a.unsqueeze(0)).shape == (1, 5, 8)
    expected = 3 * 4 + 4 + (8 + 4) * 16 + 16 + 2 * (2 * 16 + 2 * (16 * 16 + 16)) + 2 * 16 + 16 * 8 + 8
    assert mlp.num_parameters() == expected


def test_dynamics_loss_kinds():
    p = torch.randn(4, 5, 8)
    t = torch.randn(4, 5, 8)
    cos, info = dynamics_loss(p, t, "cosine")
    mse, _ = dynamics_loss(p, t, "mse")
    both, _ = dynamics_loss(p, t, "both")
    assert torch.isclose(both, cos + mse)
    assert set(info) == {"loss_cosine", "loss_mse"}


def toy_dynamics():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(3, 8, generator=g)
    return lambda z, a: z + (a @ w)[:, None, :]


def test_guided_euler_with_eta_zero_matches_plain_loop():
    target = torch.randn(1, 6, 3)
    denoise = lambda x, t: x - target  # noqa: E731
    noise = torch.randn(1, 6, 3)
    plain = guided_euler(denoise, noise, 5, None)
    proc = OGGProcessor(toy_dynamics(), torch.randn(1, 4, 8), torch.randn(1, 4, 8), 0.0, 1, 3)
    guided = guided_euler(denoise, noise, 5, proc)
    assert torch.equal(plain, guided)
    assert proc.losses == []


def test_guided_euler_applies_guidance_every_n_steps():
    target = torch.randn(1, 6, 3)
    denoise = lambda x, t: x - target  # noqa: E731
    noise = torch.randn(1, 6, 3)
    proc = OGGProcessor(toy_dynamics(), torch.randn(1, 4, 8), torch.randn(1, 4, 8), 1.0, 2, 3)
    guided = guided_euler(denoise, noise, 5, proc)
    plain = guided_euler(denoise, noise, 5, None)
    assert len(proc.losses) == 3
    assert torch.isfinite(guided).all()
    assert not torch.equal(plain, guided)


def test_ogg_step_lowers_the_guidance_loss():
    dyn = toy_dynamics()
    z = torch.randn(1, 4, 8)
    tgt = torch.randn(1, 4, 8)
    x_t = torch.randn(1, 6, 3)
    v_t = torch.randn(1, 6, 3)
    proc = OGGProcessor(dyn, z, tgt, 0.01, 1, 3)

    def loss_of(v):
        a0 = (x_t - 0.5 * v)[:, 0, :3]
        pred = dyn(z, a0)
        return float(1.0 - torch.nn.functional.cosine_similarity(pred.reshape(1, -1), tgt.reshape(1, -1)))

    v_new = proc.denoise_step(x_t, None, None, 0.5, lambda x: v_t)
    assert loss_of(v_new) < loss_of(v_t)
