import argparse
import json
from collections import deque
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla_modified.configuration_smolvla_modified import SmolVLAModifiedConfig
from lerobot.policies.smolvla_modified.modeling_smolvla_modified import SmolVLAModifiedPolicy
from lerobot.safety.siglip_dynamics_mlp import SiglipDynamicsPredictor
from lerobot.scripts.lerobot_eval_modified_detection import RobustThresholdState, robust_threshold_step
from transformers import AutoTokenizer

CAMERAS = {
    "observation.images.camera1": "observation.images.up",
    "observation.images.camera2": "observation.images.side",
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--corrector", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-repo", default="lerobot/svla_so101_pickplace")
    p.add_argument("--episodes", default="40,41,42,43,44")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cooldown", type=int, default=10)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--apply-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_policy(base: str, device: str) -> SmolVLAModifiedPolicy:
    src = PreTrainedConfig.from_pretrained(base)
    kw = {f.name: getattr(src, f.name) for f in fields(SmolVLAConfig) if f.init and hasattr(src, f.name)}
    cfg = SmolVLAModifiedConfig(**kw)
    state_shape = src.input_features["observation.state"].shape
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape)}
    for cam in CAMERAS:
        cfg.input_features[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    cfg.device = device
    policy = SmolVLAModifiedPolicy.from_pretrained(base, config=cfg)
    policy.to(device=device, dtype=torch.float32)
    policy.eval()
    return policy


def make_batch(frames, state, tokens, mask, device):
    batch = {cam: torch.from_numpy(frames[i]).to(device).float().div(255.0)[None] for i, cam in enumerate(CAMERAS)}
    batch["observation.state"] = torch.from_numpy(state).to(device).float()[None]
    batch["observation.language.tokens"] = torch.from_numpy(tokens).to(device)[None]
    batch["observation.language.attention_mask"] = torch.from_numpy(mask).to(device).bool()[None]
    return batch


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    episodes = [int(x) for x in args.episodes.split(",")]
    ds = LeRobotDataset(args.dataset_repo, root=args.dataset_root, episodes=episodes, video_backend="torchcodec")
    stats = ds.meta.stats["observation.state"]
    s_mean = np.asarray(stats["mean"], dtype=np.float32)
    s_std = np.asarray(stats["std"], dtype=np.float32)
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    fr_col = np.asarray(ds.hf_dataset["frame_index"])
    policy = build_policy(args.base, device)
    tok = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    predictor = SiglipDynamicsPredictor(args.corrector, device=device)
    policy.set_safety_predictor(predictor.model, action_dim=predictor.action_dim, eta=args.eta)
    k = int(predictor.k_step)
    frames_all, state_all, action_all, z_all = [], [], [], []
    rows = []
    tokens = mask = None
    jitter_events = 0
    for ep in episodes:
        idx = [int(i) for i in np.nonzero((ep_col == ep) & (fr_col < args.steps))[0]]
        idx.sort(key=lambda i: int(fr_col[i]))
        policy.reset()
        z_history = deque(maxlen=max(k + 1, 2))
        pred_queue = deque(maxlen=k + 2)
        th = RobustThresholdState(cooldown_steps=args.cooldown)
        for t, i in enumerate(idx):
            item = ds[i]
            if tokens is None:
                enc = tok(
                    item["task"] + "\n",
                    padding="max_length",
                    max_length=int(policy.config.tokenizer_max_length),
                    truncation=True,
                    return_tensors="np",
                )
                tokens = enc["input_ids"][0].astype(np.int64)
                mask = enc["attention_mask"][0].astype(np.int64)
            frames = np.stack(
                [(item[src].numpy() * 255.0).round().clip(0, 255).astype(np.uint8) for src in CAMERAS.values()]
            )
            state = ((item["observation.state"].numpy() - s_mean) / s_std).astype(np.float32)
            action = item["action"].numpy().astype(np.float32)
            batch = make_batch(frames, state, tokens, mask, device)
            z = np.asarray(policy.encode_siglip_only(batch), dtype=np.float32)
            z_history.append((t, z))
            score = float("nan")
            thresh = float("nan")
            trig = False
            payload = None
            if t >= k and pred_queue and pred_queue[0][0] == t:
                _, dz_pred = pred_queue[0]
                z_prev = next((hz for ht, hz in z_history if ht == t - k), None)
                if z_prev is not None:
                    dz_real = z - z_prev
                    if np.linalg.norm(dz_real) < 1e-4:
                        jitter_events += 1
                        dz_real = dz_real + np.random.randn(*dz_real.shape).astype(np.float32) * 1e-4
                    r = dz_real.reshape(-1)
                    q = dz_pred.reshape(-1)
                    denom = np.linalg.norm(r) * np.linalg.norm(q)
                    cos = float(np.dot(r, q) / denom) if denom > 0 else 0.0
                    score = 1.0 - cos
                    thresh, trig = robust_threshold_step(score, th)
                    trig = bool(trig)
                    payload = (z, dz_real, dz_pred)
                pred_queue.popleft()
            dz_next = predictor.predict_delta_z(z, action[: predictor.action_dim])
            pred_queue.append((t + k, np.asarray(dz_next, dtype=np.float32)))
            rows.append({"episode": ep, "t": t, "score": score, "threshold": float(thresh), "triggered": trig})
            if payload is not None and (trig or t == len(idx) - 1):
                np.savez(out / f"payload_ep{ep}_t{t}.npz", z=payload[0], dz_real=payload[1], dz_pred=payload[2])
            frames_all.append(frames)
            state_all.append(state)
            action_all.append(action)
            z_all.append(z)
    np.save(out / "frames.npy", np.stack(frames_all))
    np.save(out / "state.npy", np.stack(state_all))
    np.save(out / "action.npy", np.stack(action_all))
    np.save(out / "z.npy", np.stack(z_all))
    np.save(out / "tokens.npy", tokens)
    np.save(out / "mask.npy", mask)
    with open(out / "scores.json", "w") as f:
        json.dump(rows, f)
    trig_rows = [r for r in rows if r["triggered"]]
    finite = [r for r in rows if np.isfinite(r["score"])]
    pick = trig_rows[0] if trig_rows else max(finite, key=lambda r: r["score"])
    pay = (
        np.load(out / f"payload_ep{pick['episode']}_t{pick['t']}.npz")
        if (out / f"payload_ep{pick['episode']}_t{pick['t']}.npz").exists()
        else None
    )
    if pay is None:
        flat = rows.index(pick)
        ep, t = pick["episode"], pick["t"]
        z = z_all[flat]
        z_prev = z_all[flat - k]
        dz_real = z - z_prev
        dz_pred = predictor.predict_delta_z(z_all[flat - k], action_all[flat - k][: predictor.action_dim])
        np.savez(out / f"payload_ep{ep}_t{t}.npz", z=z, dz_real=dz_real, dz_pred=dz_pred)
        pay = np.load(out / f"payload_ep{ep}_t{t}.npz")
    flat = rows.index(pick)
    batch = make_batch(frames_all[flat], state_all[flat], tokens, mask, device)
    policy.reset()
    prepared = policy._prepare_batch(batch)
    images, img_masks = policy.prepare_images(prepared)
    state_t = policy.prepare_state(prepared)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(
        (1, policy.config.chunk_size, policy.config.max_action_dim), generator=gen, dtype=torch.float32
    ).to(device)
    common = (
        images,
        img_masks,
        prepared["observation.language.tokens"],
        prepared["observation.language.attention_mask"],
        state_t,
    )
    guided = policy.model.sample_actions(
        *common,
        noise=noise.clone(),
        guidance_replan=True,
        guidance_current_z_t=pay["z"],
        guidance_current_delta_z_t=pay["dz_real"],
        guidance_current_delta_z_right=pay["dz_pred"],
        guidance_apply_every=args.apply_every,
        guidance_loss_objective="attract_delta_z_correction",
        guidance_eta=args.eta,
    )
    gstats = policy.model.get_last_guidance_stats()
    with torch.no_grad():
        plain = policy.model.sample_actions(*common, noise=noise.clone())
    adim = int(policy.config.output_features["action"].shape[0])
    np.savez(
        out / "ogg.npz",
        noise=noise.cpu().numpy(),
        guided=guided[:, :, :adim].detach().cpu().numpy(),
        plain=plain[:, :, :adim].detach().cpu().numpy(),
        flat_index=flat,
        z=pay["z"],
        dz_real=pay["dz_real"],
        dz_pred=pay["dz_pred"],
    )
    summary = {
        "device": device,
        "torch": torch.__version__,
        "k_step": k,
        "action_dim": int(predictor.action_dim),
        "episodes": episodes,
        "steps": len(rows),
        "finite_scores": len(finite),
        "triggers": [(r["episode"], r["t"]) for r in trig_rows],
        "jitter_events": jitter_events,
        "ogg_from": [pick["episode"], pick["t"], bool(pick["triggered"])],
        "ogg_stats": {k2: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k2, v in gstats.items()},
        "guided_minus_plain_max_abs": float((guided[:, :, :adim] - plain[:, :, :adim]).abs().max()),
        "task": ds[0]["task"],
    }
    with open(out / "summary_upstream.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k2: summary[k2] for k2 in ("steps", "finite_scores", "triggers", "guided_minus_plain_max_abs")}))


if __name__ == "__main__":
    main()
