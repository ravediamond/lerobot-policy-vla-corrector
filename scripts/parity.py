import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class

from lerobot_policy_vla_corrector import VLACorrectorConfig, VLACorrectorPolicy
from lerobot_policy_vla_corrector.corrector import Monitor, OGGProcessor
from lerobot_policy_vla_corrector.upstream_checkpoint import load_upstream_corrector

CAMERAS = ("observation.images.camera1", "observation.images.camera2")
BARS = {"C1": 1e-4, "C3": 1e-3}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--corrector", required=True)
    p.add_argument("--upstream-out", required=True)
    p.add_argument("--cooldown", type=int, default=10)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--apply-every", type=int, default=3)
    p.add_argument("--c4-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_base(path: str, device: str):
    cfg = PreTrainedConfig.from_pretrained(path)
    state_shape = cfg.input_features["observation.state"].shape
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape)}
    for cam in CAMERAS:
        cfg.input_features[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    cfg.device = device
    base = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    base.to(device=device, dtype=torch.float32)
    base.eval()
    return base


def make_batch(frames, state, tokens, mask, device):
    batch = {cam: torch.from_numpy(frames[i]).to(device).float().div(255.0)[None] for i, cam in enumerate(CAMERAS)}
    batch["observation.state"] = torch.from_numpy(state).to(device).float()[None]
    batch["observation.language.tokens"] = torch.from_numpy(tokens).to(device)[None]
    batch["observation.language.attention_mask"] = torch.from_numpy(mask).to(device).bool()[None]
    return batch


def build_policy(base, up, mode: str, args, device: str) -> VLACorrectorPolicy:
    cfg = VLACorrectorConfig(
        mode=mode,
        k_step=up.k_step,
        token_dim=up.token_dim,
        mlp_widths=up.widths,
        action_embed_dim=up.action_embed_dim,
        dropout=up.dropout,
        cooldown=args.cooldown,
        ogg_eta=args.eta,
        ogg_apply_every=args.apply_every,
        device=device,
    )
    policy = VLACorrectorPolicy(cfg, base=base)
    policy.corrector.load_state_dict(up.state_dict, strict=True)
    policy.corrector.to(device=device, dtype=torch.float32)
    policy.eval()
    return policy


def run_monitor(policy, latents_fn, n_eps, steps, action, device):
    scores, triggers = [], []
    for e in range(n_eps):
        mon: Monitor = policy.monitor
        mon.reset()
        policy.base.reset()
        for t in range(steps):
            flat = e * steps + t
            z = latents_fn(flat)
            res = mon.observe(t, z)
            scores.append(res.score)
            triggers.append(bool(res.triggered))
            mon.predict(t, z, torch.from_numpy(action[flat][: policy.action_dim]).to(device))
    return np.asarray(scores, dtype=np.float64), np.asarray(triggers)


def max_abs(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    up_dir = Path(args.upstream_out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    summary_up = json.loads((up_dir / "summary_upstream.json").read_text())
    rows = json.loads((up_dir / "scores.json").read_text())
    frames = np.load(up_dir / "frames.npy", mmap_mode="r")
    state = np.load(up_dir / "state.npy")
    action = np.load(up_dir / "action.npy")
    z_up = np.load(up_dir / "z.npy", mmap_mode="r")
    tokens = np.load(up_dir / "tokens.npy")
    mask = np.load(up_dir / "mask.npy")
    n_eps = len(summary_up["episodes"])
    steps = len(rows) // n_eps
    up_scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    up_trig = np.asarray([bool(r["triggered"]) for r in rows])
    up = load_upstream_corrector(args.corrector)
    base = load_base(args.base, device)
    policy = build_policy(base, up, "monitor_only", args, device)
    report = {"device": device, "torch": torch.__version__, "k_step": up.k_step, "steps": len(rows)}

    z_plugin = []

    def own_latents(flat):
        batch = make_batch(frames[flat], state[flat], tokens, mask, device)
        z, _ = policy.backend.latents(batch)
        z_plugin.append(z[0].detach().cpu().numpy())
        return z[0]

    def shared_latents(flat):
        return torch.from_numpy(np.ascontiguousarray(z_up[flat])).to(device)

    s_own, t_own = run_monitor(policy, own_latents, n_eps, steps, action, device)
    s_shared, t_shared = run_monitor(policy, shared_latents, n_eps, steps, action, device)
    z_plugin = np.stack(z_plugin)
    report["latent_max_abs"] = float(np.max(np.abs(z_plugin - z_up)))
    report["latent_max_rel"] = float(np.max(np.abs(z_plugin - z_up)) / np.max(np.abs(z_up)))
    report["C1_end_to_end_max_abs"] = max_abs(s_own, up_scores)
    report["C1_shared_latents_max_abs"] = max_abs(s_shared, up_scores)
    report["finite_scores"] = int(np.isfinite(up_scores).sum())
    report["triggers_upstream"] = [int(i) for i in np.nonzero(up_trig)[0]]
    report["triggers_plugin_end_to_end"] = [int(i) for i in np.nonzero(t_own)[0]]
    report["triggers_plugin_shared"] = [int(i) for i in np.nonzero(t_shared)[0]]
    report["C2_end_to_end_identical"] = bool(np.array_equal(t_own, up_trig))
    report["C2_shared_identical"] = bool(np.array_equal(t_shared, up_trig))
    np.savez(out / "scores_plugin.npz", own=s_own, shared=s_shared, upstream=up_scores)

    ogg = np.load(up_dir / "ogg.npz")
    flat = int(ogg["flat_index"])
    noise = torch.from_numpy(ogg["noise"]).to(device)
    z_t = torch.from_numpy(ogg["z"]).to(device)[None]
    target = torch.from_numpy(ogg["dz_pred"] - ogg["dz_real"]).to(device)[None]
    batch = make_batch(frames[flat], state[flat], tokens, mask, device)
    policy.base.reset()
    proc = OGGProcessor(
        policy.corrector, z_t, target, eta=args.eta, apply_every=args.apply_every, action_dim=policy.action_dim
    )
    guided = policy.backend.guided_chunk(batch, noise.clone(), proc, None).detach().cpu().numpy()
    policy.base.reset()
    plain = policy.backend.guided_chunk(batch, noise.clone(), None, None).detach().cpu().numpy()
    report["C3_guided_max_abs"] = float(np.max(np.abs(guided - ogg["guided"])))
    report["C3_plain_max_abs"] = float(np.max(np.abs(plain - ogg["plain"])))
    report["ogg_effect_upstream_max_abs"] = float(np.max(np.abs(ogg["guided"] - ogg["plain"])))
    report["ogg_effect_plugin_max_abs"] = float(np.max(np.abs(guided - plain)))
    report["ogg_applied_steps_plugin"] = len(proc.losses)
    report["ogg_from"] = summary_up["ogg_from"]

    policy = None
    base = None
    torch.cuda.empty_cache()
    ref = load_base(args.base, device)
    wrapped = build_policy(load_base(args.base, device), up, "off", args, device)
    ref.reset()
    wrapped.reset()
    worst = 0.0
    identical = True
    for i in range(args.c4_steps):
        flat_i = i % len(rows)
        b = make_batch(frames[flat_i], state[flat_i], tokens, mask, device)
        g = torch.Generator(device="cpu").manual_seed(args.seed + i)
        nz = torch.randn((1, ref.config.chunk_size, ref.config.max_action_dim), generator=g).to(device)
        a_ref = ref.select_action(dict(b), noise=nz.clone())
        a_wrap = wrapped.select_action(dict(b), noise=nz.clone())
        identical = identical and torch.equal(a_ref, a_wrap)
        worst = max(worst, float((a_ref - a_wrap).abs().max()))
    report["C4_steps"] = args.c4_steps
    report["C4_max_abs"] = worst
    report["C4_bit_identical"] = identical
    report["bars"] = {
        "C1": report["C1_end_to_end_max_abs"] <= BARS["C1"],
        "C1_shared": report["C1_shared_latents_max_abs"] <= BARS["C1"],
        "C2": report["C2_end_to_end_identical"],
        "C3": report["C3_guided_max_abs"] <= BARS["C3"],
        "C4": identical and worst == 0.0,
    }
    with open(out / "parity_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["bars"]))


if __name__ == "__main__":
    main()
