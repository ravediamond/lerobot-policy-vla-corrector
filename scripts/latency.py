import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class

from lerobot_policy_vla_corrector import VLACorrectorConfig, VLACorrectorPolicy
from lerobot_policy_vla_corrector.upstream_checkpoint import load_upstream_corrector

CAMERAS = ("observation.images.camera1", "observation.images.camera2")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--corrector", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtypes", default="fp32,bf16")
    p.add_argument("--compile", default="eager,compile")
    p.add_argument("--modes", default="off,monitor_only,truncate_ogg")
    p.add_argument("--every", default="1,2")
    p.add_argument("--reuse", default="1")
    p.add_argument("--ogg-reps", type=int, default=20)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_base(path, device, dtype):
    cfg = PreTrainedConfig.from_pretrained(path)
    state_shape = cfg.input_features["observation.state"].shape
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape)}
    for cam in CAMERAS:
        cfg.input_features[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    cfg.device = device
    base = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    base.to(device=device, dtype=dtype)
    base.eval()
    return base


def make_batch(frames, state, tokens, mask, device):
    batch = {cam: torch.from_numpy(frames[i]).to(device).float().div(255.0)[None] for i, cam in enumerate(CAMERAS)}
    batch["observation.state"] = torch.from_numpy(state).to(device).float()[None]
    batch["observation.language.tokens"] = torch.from_numpy(tokens).to(device)[None]
    batch["observation.language.attention_mask"] = torch.from_numpy(mask).to(device).bool()[None]
    return batch


def percentile(xs, q):
    return float(np.percentile(np.asarray(xs), q)) if xs else float("nan")


def run_config(base, up, data, device, mode, every, reuse, steps, warmup, compiled):
    cfg = VLACorrectorConfig(
        mode=mode,
        k_step=up.k_step,
        token_dim=up.token_dim,
        mlp_widths=up.widths,
        action_embed_dim=up.action_embed_dim,
        monitor_every_n_steps=every,
        reuse_vision_features=reuse,
        device=device,
    )
    policy = VLACorrectorPolicy(cfg, base=base)
    policy.corrector.load_state_dict(up.state_dict, strict=True)
    policy.corrector.to(device=device, dtype=torch.float32)
    if compiled:
        policy.corrector.forward = torch.compile(policy.corrector.forward)
    policy.eval()
    frames, state, tokens, mask = data
    n = len(frames)
    times, replans, triggers, monitor_ms = [], 0, 0, []
    replan_times, plain_times = [], []
    policy.reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + warmup):
        batch = make_batch(frames[i % n], state[i % n], tokens, mask, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.select_action(batch)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i < warmup:
            continue
        times.append(dt)
        info = policy.last_info if mode != "off" else {}
        if info.get("replanned"):
            replans += 1
            replan_times.append(dt)
        else:
            plain_times.append(dt)
        if info.get("triggered"):
            triggers += 1
        if info and np.isfinite(info.get("monitor_ms", float("nan"))):
            monitor_ms.append(float(info["monitor_ms"]))
    peak = torch.cuda.max_memory_allocated() / 2**20
    if mode == "off":
        replans = int(np.ceil(steps / base.config.n_action_steps))
    return {
        "mode": mode,
        "every": every,
        "reuse_vision_features": reuse,
        "steps": len(times),
        "median_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "max_ms": float(max(times)),
        "median_non_replan_ms": percentile(plain_times, 50),
        "p95_non_replan_ms": percentile(plain_times, 95),
        "median_replan_ms": percentile(replan_times, 50),
        "replans": replans,
        "triggers": triggers,
        "monitor_median_ms": percentile(monitor_ms, 50),
        "monitor_p95_ms": percentile(monitor_ms, 95),
        "peak_mem_mb": peak,
    }


def ogg_cost(base, up, data, device, reps):
    from lerobot_policy_vla_corrector.corrector import OGGProcessor

    cfg = VLACorrectorConfig(
        mode="truncate_ogg", k_step=up.k_step, token_dim=up.token_dim, mlp_widths=up.widths, device=device
    )
    policy = VLACorrectorPolicy(cfg, base=base)
    policy.corrector.load_state_dict(up.state_dict, strict=True)
    policy.corrector.to(device=device, dtype=torch.float32)
    frames, state, tokens, mask = data
    batch = make_batch(frames[0], state[0], tokens, mask, device)
    z, _ = policy.backend.latents(batch)
    target = torch.randn_like(z)
    out = {}
    for name in ("plain", "guided"):
        ts = []
        for r in range(reps + 3):
            policy.base.reset()
            proc = (
                OGGProcessor(policy.corrector, z, target, cfg.ogg_eta, cfg.ogg_apply_every, policy.action_dim)
                if name == "guided"
                else None
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.backend.guided_chunk(dict(batch), None, proc, None)
            torch.cuda.synchronize()
            if r >= 3:
                ts.append((time.perf_counter() - t0) * 1000.0)
        out[f"{name}_chunk_median_ms"] = percentile(ts, 50)
    out["ogg_extra_ms"] = out["guided_chunk_median_ms"] - out["plain_chunk_median_ms"]
    return out


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    fd = Path(args.frames_dir)
    data = (
        np.load(fd / "frames.npy", mmap_mode="r"),
        np.load(fd / "state.npy"),
        np.load(fd / "tokens.npy"),
        np.load(fd / "mask.npy"),
    )
    up = load_upstream_corrector(args.corrector)
    results = []
    path = out / "latency.json"
    for dname in args.dtypes.split(","):
        base = load_base(args.base, device, DTYPES[dname])
        for comp in args.compile.split(","):
            compiled = comp == "compile"
            if compiled:
                vwe = base.model.vlm_with_expert
                vwe.__dict__["embed_image"] = torch.compile(vwe.embed_image)
            for mode in args.modes.split(","):
                everies = [1] if mode == "off" else [int(x) for x in args.every.split(",")]
                for every in everies:
                    for reuse in [bool(int(x)) for x in args.reuse.split(",")]:
                        if mode == "off" and not reuse:
                            continue
                        row = {"dtype": dname, "compile": comp}
                        try:
                            row.update(
                                run_config(
                                    base, up, data, device, mode, every, reuse, args.steps, args.warmup, compiled
                                )
                            )
                        except Exception as err:
                            row.update(mode=mode, every=every, error=f"{type(err).__name__}: {str(err)[:300]}")
                        results.append(row)
                        print(json.dumps(row), flush=True)
                        path.write_text(json.dumps({"rows": results}, indent=2))
            if compiled:
                base.model.vlm_with_expert.__dict__.pop("embed_image", None)
        if "eager" in args.compile.split(","):
            row = {"dtype": dname, "compile": "eager", "mode": "ogg_cost"}
            row.update(ogg_cost(base, up, data, device, args.ogg_reps))
            results.append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps({"rows": results}, indent=2))
        del base
        torch.cuda.empty_cache()
    meta = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "steps": args.steps,
        "warmup": args.warmup,
    }
    path.write_text(json.dumps({"meta": meta, "rows": results}, indent=2))


if __name__ == "__main__":
    main()
