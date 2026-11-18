from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

SCALE_WIDTHS = {
    "4m": (1024, 1024, 1024),
    "20m": (2048, 2048, 2048, 2048),
    "100m": (4096, 4096, 4096, 4096, 4096, 4096),
}


@dataclass
class UpstreamCorrector:
    state_dict: dict[str, Tensor]
    token_dim: int
    action_dim: int
    widths: tuple[int, ...]
    action_embed_dim: int
    dropout: float
    k_step: int
    checkpoint: Path


def _pick_checkpoint(model_dir: Path) -> Path:
    for name in ("best_model.pt", "best.pt", "final_model.pt", "last.pt"):
        if (model_dir / name).is_file():
            return model_dir / name
    found = sorted(model_dir.glob("best_*.pt")) or sorted(model_dir.glob("*.pt"))
    if not found:
        raise FileNotFoundError(f"no .pt checkpoint in {model_dir}")
    return found[0]


def load_upstream_corrector(path: str | Path) -> UpstreamCorrector:
    path = Path(path)
    model_dir = path.parent if path.is_file() else path
    ckpt = path if path.is_file() else _pick_checkpoint(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("model_type", cfg.get("model_type", "mlp"))).lower()
    if model_type != "mlp":
        raise ValueError(f"only the residual MLP corrector is supported, got model_type={model_type!r}")
    scale = str(model_cfg.get("scale", "20m")).lower()
    widths = SCALE_WIDTHS.get(scale) or tuple(int(x) for x in model_cfg.get("custom_widths", (2048,) * 4))
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    k_step = int(cfg.get("k_step", 1))
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        train_cfg = obj.get("train_cfg", {})
        if isinstance(train_cfg, dict):
            k_step = int(train_cfg.get("k_step", k_step))
        k_step = int(obj.get("k_step", k_step))
        if int(obj.get("h_window", 1)) != 1:
            raise ValueError("only h_window=1 checkpoints are supported")
    else:
        state = obj
    token_dim = int(model_cfg["token_dim"])
    state = {k: v.detach().to(torch.float32) for k, v in state.items()}
    state.setdefault("latent_mean", torch.zeros(token_dim))
    state.setdefault("latent_std", torch.ones(token_dim))
    return UpstreamCorrector(
        state_dict=state,
        token_dim=token_dim,
        action_dim=int(model_cfg["action_dim"]),
        widths=tuple(widths),
        action_embed_dim=int(model_cfg.get("action_embed_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        k_step=k_step,
        checkpoint=ckpt,
    )
