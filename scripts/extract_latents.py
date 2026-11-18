import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from lerobot_policy_vla_corrector.backends import make_backend

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse():
    p = argparse.ArgumentParser(description="LeRobot dataset -> per-episode latent shards for the corrector")
    p.add_argument("--base", required=True, help="pretrained flow-matching policy (path or Hub id)")
    p.add_argument("--backend", default="smolvla")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--episodes", default="", help="comma list or a-b range; empty = all")
    p.add_argument("--rename", default="", help="dataset_key=policy_key pairs, comma separated")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    p.add_argument("--video-backend", default=None)
    p.add_argument("--min-free-gb", type=float, default=5.0)
    p.add_argument(
        "--drop-episode-files", action="store_true", help="delete the cached dataset files after each episode"
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_episodes(spec: str, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rename = dict(pair.split("=") for pair in args.rename.split(",") if pair)
    cfg = PreTrainedConfig.from_pretrained(args.base)
    cfg.device = device
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    if rename:
        feats = {k: v for k, v in cfg.input_features.items() if v.type != FeatureType.VISUAL}
        for src, dst in rename.items():
            shape = tuple(meta.features[src]["shape"])
            if meta.features[src]["dtype"] in ("video", "image") and shape[-1] == 3:
                shape = (shape[2], shape[0], shape[1])
            feats[dst] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
        cfg.input_features = feats
    base = get_policy_class(cfg.type).from_pretrained(args.base, config=cfg)
    base.to(device=device, dtype=DTYPES[args.dtype])
    base.eval()
    backend = make_backend(args.backend, base)
    pre, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.base,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename},
        },
    )
    episodes = parse_episodes(args.episodes, meta.total_episodes)
    action_dim = int(cfg.output_features["action"].shape[0])
    for ep in episodes:
        shard = out / f"episode_{ep:06d}.npz"
        if shard.exists():
            continue
        free_gb = shutil.disk_usage(out).free / 2**30
        if free_gb < args.min_free_gb:
            raise SystemExit(f"free disk {free_gb:.1f} GB < {args.min_free_gb} GB; stopping before episode {ep}")
        ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[ep], video_backend=args.video_backend)
        n = len(ds) if args.max_frames <= 0 else min(len(ds), args.max_frames)
        zs, acts, frames = [], [], []
        for i in range(0, n, args.stride):
            item = ds[i]
            batch = pre(dict(item))
            with torch.no_grad():
                z, _ = backend.latents(batch)
            zs.append(z[0].to(torch.float16).cpu().numpy())
            acts.append(batch["action"].reshape(-1)[:action_dim].float().cpu().numpy())
            frames.append(int(item["frame_index"]))
        np.savez(
            shard,
            z=np.stack(zs),
            action=np.stack(acts).astype(np.float32),
            frame_index=np.asarray(frames, dtype=np.int64),
            episode_index=np.full(len(frames), ep, dtype=np.int64),
        )
        print(json.dumps({"episode": ep, "frames": len(frames), "shape": list(np.stack(zs).shape)}), flush=True)
        if args.drop_episode_files and args.root is None:
            for f in ds.root.rglob("*"):
                if f.is_file() and f.suffix in (".mp4", ".parquet") and "meta" not in f.parts:
                    f.unlink()
    info = {
        "base": args.base,
        "backend": args.backend,
        "repo_id": args.repo_id,
        "episodes": episodes,
        "stride": args.stride,
        "dtype": args.dtype,
        "fps": meta.fps,
        "rename": rename,
        "action_space": "normalised by the base policy preprocessor",
    }
    (out / "latents_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
