import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from lerobot_policy_vla_corrector.corrector import LatentDynamicsMLP, dynamics_loss


class Pairs(Dataset):
    def __init__(self, shards: list[Path], k_step: int, stride: int):
        self.z = []
        self.a = []
        self.index = []
        hop = k_step // stride
        if hop < 1 or k_step % stride != 0:
            raise ValueError("k_step must be a positive multiple of the extraction stride")
        for s, path in enumerate(shards):
            d = np.load(path)
            self.z.append(d["z"])
            self.a.append(d["action"])
            for t in range(len(d["z"]) - hop):
                self.index.append((s, t, t + hop))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, t, u = self.index[i]
        z_t = torch.from_numpy(self.z[s][t].astype(np.float32))
        z_u = torch.from_numpy(self.z[s][u].astype(np.float32))
        return z_t, torch.from_numpy(self.a[s][t]), z_u - z_t


def parse():
    p = argparse.ArgumentParser(description="train the latent dynamics corrector on latent shards")
    p.add_argument("--latents", required=True)
    p.add_argument("--base", required=True, help="base policy path written into the corrector config")
    p.add_argument("--k-step", type=int, default=10)
    p.add_argument("--widths", default="2048,2048,2048,2048")
    p.add_argument("--action-embed-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss", default="both", choices=["cosine", "mse", "both"])
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--normalize-latents", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def evaluate(model, loader, device, kind):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for z, a, dz in loader:
            z, a, dz = z.to(device), a.to(device), dz.to(device)
            loss, _ = dynamics_loss(model(z, a), dz, kind)
            tot += float(loss) * len(z)
            n += len(z)
    model.train()
    return tot / max(n, 1)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lat = Path(args.latents)
    info = json.loads((lat / "latents_info.json").read_text())
    shards = sorted(lat.glob("episode_*.npz"))
    if len(shards) < 2:
        raise SystemExit("need at least two episode shards for a train / validation split")
    random.Random(args.seed).shuffle(shards)
    n_val = max(1, int(round(len(shards) * args.val_ratio)))
    train_set = Pairs(shards[n_val:], args.k_step, int(info.get("stride", 1)))
    val_set = Pairs(shards[:n_val], args.k_step, int(info.get("stride", 1)))
    z0, a0, _ = train_set[0]
    widths = tuple(int(x) for x in args.widths.split(","))
    model = LatentDynamicsMLP(
        token_dim=int(z0.shape[-1]),
        action_dim=int(a0.shape[-1]),
        widths=widths,
        action_embed_dim=args.action_embed_dim,
        normalize_latents=args.normalize_latents,
    ).to(device)
    if args.normalize_latents:
        sample = torch.stack([train_set[i][0] for i in range(0, len(train_set), max(1, len(train_set) // 512))])
        model.latent_mean.copy_(sample.reshape(-1, sample.shape[-1]).mean(0).to(device))
        model.latent_std.copy_(sample.reshape(-1, sample.shape[-1]).std(0).clamp_min(1e-4).to(device))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        run, seen = 0.0, 0
        for z, a, dz in train_loader:
            z, a, dz = z.to(device), a.to(device), dz.to(device)
            loss, _ = dynamics_loss(model(z, a), dz, args.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            run += float(loss) * len(z)
            seen += len(z)
        val = evaluate(model, val_loader, device, args.loss)
        history.append({"epoch": epoch, "train": run / max(seen, 1), "val": val})
        print(json.dumps(history[-1]), flush=True)
        if val < best:
            best = val
            state = {f"corrector.{k}": v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "type": "vla_corrector",
        "base_policy_path": args.base,
        "encoder_backend": info.get("backend", "smolvla"),
        "k_step": args.k_step,
        "token_dim": int(z0.shape[-1]),
        "mlp_widths": list(widths),
        "action_embed_dim": args.action_embed_dim,
        "normalize_latents": bool(args.normalize_latents),
        "train_loss": args.loss,
    }
    (out / "corrector_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "train_log.json").write_text(
        json.dumps(
            {"best_val": best, "history": history, "train_pairs": len(train_set), "val_pairs": len(val_set)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
