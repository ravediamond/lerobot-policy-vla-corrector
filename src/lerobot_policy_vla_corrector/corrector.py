from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class LatentDynamicsMLP(nn.Module):
    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        widths: tuple[int, ...] = (2048, 2048, 2048, 2048),
        action_embed_dim: int = 256,
        dropout: float = 0.0,
        normalize_latents: bool = False,
    ):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths needs at least two entries")
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.normalize_latents = normalize_latents
        self.action_proj = nn.Linear(action_dim, action_embed_dim)
        self.in_proj = nn.Linear(token_dim + action_embed_dim, widths[0])
        self.blocks = nn.ModuleList([ResidualBlock(w, dropout) for w in widths])
        transitions = []
        for a, b in zip(widths[:-1], widths[1:], strict=True):
            if a != b:
                transitions.append(nn.Sequential(nn.LayerNorm(a), nn.Linear(a, b), nn.SiLU()))
            else:
                transitions.append(nn.Identity())
        self.transitions = nn.ModuleList(transitions)
        self.out_norm = nn.LayerNorm(widths[-1])
        self.out_proj = nn.Linear(widths[-1], token_dim)
        self.register_buffer("latent_mean", torch.zeros(token_dim))
        self.register_buffer("latent_std", torch.ones(token_dim))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z_t: Tensor, a_t: Tensor) -> Tensor:
        squeeze = False
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(0)
            squeeze = True
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(0)
        if z_t.ndim != 3 or a_t.ndim != 2 or z_t.shape[0] != a_t.shape[0]:
            raise ValueError(f"expected z_t (B,L,D) and a_t (B,A), got {tuple(z_t.shape)} and {tuple(a_t.shape)}")
        if z_t.shape[-1] != self.token_dim or a_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"token_dim {z_t.shape[-1]} vs {self.token_dim}, action_dim {a_t.shape[-1]} vs {self.action_dim}"
            )
        if self.normalize_latents:
            z_t = (z_t - self.latent_mean) / self.latent_std
        bsz, lsz, _ = z_t.shape
        a_emb = self.action_proj(a_t)[:, None, :].expand(bsz, lsz, -1)
        x = self.in_proj(torch.cat([z_t, a_emb], dim=-1))
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        delta = self.out_proj(self.out_norm(x))
        if self.normalize_latents:
            delta = delta * self.latent_std
        return delta[0] if squeeze else delta


def cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return 1.0 - F.cosine_similarity(p, t, dim=-1).mean()


def dynamics_loss(pred: Tensor, target: Tensor, kind: str) -> tuple[Tensor, dict[str, float]]:
    cos = cosine_loss(pred, target)
    mse = F.mse_loss(pred, target)
    if kind == "cosine":
        total = cos
    elif kind == "mse":
        total = mse
    elif kind == "both":
        total = mse + cos
    else:
        raise ValueError(f"unknown loss {kind!r}")
    return total, {"loss_cosine": float(cos.detach()), "loss_mse": float(mse.detach())}


def mismatch_score(delta_real: Tensor, delta_pred: Tensor) -> float:
    r = delta_real.detach().reshape(-1).to(torch.float32)
    p = delta_pred.detach().reshape(-1).to(torch.float32)
    denom = float(r.norm()) * float(p.norm())
    cos = float(torch.dot(r, p)) / denom if denom > 0 else 0.0
    return 1.0 - cos


@dataclass
class ThresholdConfig:
    window: int = 15
    noise_floor: float = 0.25
    bootstrap_threshold: float = 10.0
    bootstrap_size: int = 15
    ewma_alpha: float = 0.35
    lambda_on: float = 3.0
    lambda_off: float = 2.0
    trigger_margin: float = 0.02
    jump_trigger: float = 0.06
    patience: int = 5
    reset_steps: int = 5
    cooldown: int = 10
    hard_retrigger_margin: float = 0.08
    hard_retrigger_steps: int = 3


class ThresholdState:
    def __init__(self, cfg: ThresholdConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.bootstrap_queue: list[float] = []
        self.bootstrapped = False
        self.clean_history: list[float] = []
        self.ewma: float | None = None
        self.alert = False
        self.over_count = 0
        self.under_count = 0
        self.hard_count = 0
        self.cooldown_remaining = 0
        self.threshold = float("nan")
        self.release = float("nan")

    def _bands(self, values: list[float]) -> tuple[float, float]:
        q = np.asarray(values, dtype=np.float32)
        med = float(np.median(q))
        mad = float(np.median(np.abs(q - med)))
        mad = max(mad, 1e-6)
        ton = max(self.cfg.noise_floor, med + float(self.cfg.lambda_on) * mad)
        toff = max(self.cfg.noise_floor, med + float(self.cfg.lambda_off) * mad)
        if toff > ton:
            toff = ton
        return ton, toff

    def step(self, score: float) -> tuple[float, bool]:
        cfg = self.cfg
        if not math.isfinite(score):
            self.threshold = float("nan")
            self.release = float("nan")
            return float("nan"), False
        if self.ewma is None or not math.isfinite(self.ewma):
            ebar = float(score)
            jump = 0.0
        else:
            alpha = float(min(max(cfg.ewma_alpha, 1e-6), 1.0))
            ebar = alpha * float(score) + (1.0 - alpha) * float(self.ewma)
            jump = abs(ebar - float(self.ewma))
        self.ewma = float(ebar)
        if not self.bootstrapped:
            thresh = float(cfg.bootstrap_threshold)
            self.bootstrap_queue.append(float(ebar))
            if len(self.bootstrap_queue) > int(cfg.bootstrap_size):
                self.bootstrap_queue = self.bootstrap_queue[-int(cfg.bootstrap_size) :]
            if len(self.bootstrap_queue) >= int(cfg.bootstrap_size):
                ton, toff = self._bands(self.bootstrap_queue)
                thresh = ton
                self.bootstrapped = True
                q = np.asarray(self.bootstrap_queue, dtype=np.float32)
                self.clean_history = [float(x) for x in q.tolist()]
                self.release = float(toff)
            else:
                self.release = thresh
            self.threshold = thresh
            return thresh, False
        ton, toff = self._bands(self.clean_history[-max(2, int(cfg.window)) :])
        self.threshold = float(ton)
        self.release = float(toff)
        cond_over = ebar > (ton + float(cfg.trigger_margin))
        cond_jump = jump > float(cfg.jump_trigger) and ebar > ton
        cond_hard = ebar > (ton + float(cfg.hard_retrigger_margin))
        self.over_count = self.over_count + 1 if cond_over else 0
        self.hard_count = self.hard_count + 1 if cond_hard else 0
        self.under_count = self.under_count + 1 if ebar < toff else 0
        if self.alert and self.under_count >= int(cfg.reset_steps):
            self.alert = False
            self.over_count = 0
            self.hard_count = 0
        triggered = False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.hard_count >= int(cfg.hard_retrigger_steps):
                triggered = True
        elif cond_jump or self.over_count >= int(cfg.patience):
            triggered = True
        if triggered:
            self.alert = True
            self.cooldown_remaining = max(0, int(cfg.cooldown))
            self.over_count = 0
            self.hard_count = 0
            self.under_count = 0
        self.clean_history.append(float(min(ebar, ton)) if self.alert else float(ebar))
        if len(self.clean_history) > 4 * max(2, int(cfg.window)):
            self.clean_history = self.clean_history[-2 * max(2, int(cfg.window)) :]
        return float(ton), bool(triggered)


@dataclass
class MonitorStep:
    step: int
    score: float
    threshold: float
    release: float
    triggered: bool
    delta_real: Tensor | None = None
    delta_pred: Tensor | None = None


class Monitor:
    def __init__(
        self,
        dynamics: Callable[[Tensor, Tensor], Tensor],
        k_step: int,
        thresholds: ThresholdConfig,
        every_n: int = 1,
        jitter_seed: int = 0,
    ):
        if k_step < 1 or every_n < 1 or k_step % every_n != 0:
            raise ValueError("k_step must be a positive multiple of every_n")
        self.dynamics = dynamics
        self.k_step = k_step
        self.every_n = every_n
        self.jitter_seed = jitter_seed
        self.state = ThresholdState(thresholds)
        self.reset()

    def reset(self) -> None:
        self.state.reset()
        self.history: dict[int, Tensor] = {}
        self.queue: deque[tuple[int, Tensor]] = deque()
        self.generator = torch.Generator().manual_seed(self.jitter_seed)
        self.triggers: list[int] = []

    def active(self, step: int) -> bool:
        return step % self.every_n == 0

    def observe(self, step: int, z_t: Tensor) -> MonitorStep:
        self.history[step] = z_t.detach()
        for old in [s for s in self.history if s < step - self.k_step]:
            del self.history[old]
        while self.queue and self.queue[0][0] < step:
            self.queue.popleft()
        score = float("nan")
        triggered = False
        delta_real = None
        delta_pred = None
        if self.queue and self.queue[0][0] == step:
            _, delta_pred = self.queue.popleft()
            z_prev = self.history.get(step - self.k_step)
            if z_prev is not None:
                delta_real = z_t.detach() - z_prev
                if float(delta_real.norm()) < 1e-4:
                    noise = torch.randn(delta_real.shape, generator=self.generator, dtype=torch.float32)
                    delta_real = delta_real + noise.to(delta_real) * 1e-4
                score = mismatch_score(delta_real, delta_pred)
                _, triggered = self.state.step(score)
                if triggered:
                    self.triggers.append(step)
        return MonitorStep(
            step=step,
            score=score,
            threshold=self.state.threshold,
            release=self.state.release,
            triggered=triggered,
            delta_real=delta_real,
            delta_pred=delta_pred,
        )

    @torch.no_grad()
    def predict(self, step: int, z_t: Tensor, action: Tensor) -> Tensor:
        pred = self.dynamics(z_t.detach(), action.detach().to(z_t)).detach()
        self.queue.append((step + self.k_step, pred))
        return pred


def ogg_velocity(
    x_t: Tensor,
    v_t: Tensor,
    time: float | Tensor,
    dynamics: Callable[[Tensor, Tensor], Tensor],
    z_t: Tensor,
    target_delta: Tensor,
    eta: float,
    action_dim: int,
) -> tuple[Tensor, float, float]:
    bsize = x_t.shape[0]
    v_in = v_t.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        t = torch.as_tensor(time, dtype=torch.float32, device=x_t.device).reshape(-1, 1, 1)
        x0_hat = x_t.detach() - t * v_in
        a0 = x0_hat[:, 0, :action_dim]
        z = z_t if z_t.ndim == 3 else z_t.unsqueeze(0)
        if z.shape[0] == 1 and bsize > 1:
            z = z.expand(bsize, -1, -1)
        tgt = target_delta if target_delta.ndim == 3 else target_delta.unsqueeze(0)
        if tgt.shape[0] == 1 and bsize > 1:
            tgt = tgt.expand(bsize, -1, -1)
        pred = dynamics(z, a0)
        loss = 1.0 - F.cosine_similarity(pred.reshape(bsize, -1), tgt.reshape(bsize, -1), dim=-1).mean()
        grad = torch.autograd.grad(loss, v_in)[0]
    v_new = (v_in - eta * grad).detach()
    return v_new, float(loss.detach()), float(grad.detach().norm())


class OGGProcessor:
    def __init__(
        self,
        dynamics: Callable[[Tensor, Tensor], Tensor],
        z_t: Tensor,
        target_delta: Tensor,
        eta: float,
        apply_every: int,
        action_dim: int,
    ):
        self.dynamics = dynamics
        self.z_t = z_t
        self.target_delta = target_delta
        self.eta = eta
        self.apply_every = max(1, int(apply_every))
        self.action_dim = action_dim
        self.step = 0
        self.losses: list[float] = []
        self.grad_norms: list[float] = []

    def is_debug_enabled(self) -> bool:
        return False

    def track(self, **kwargs) -> None:
        return None

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial: Callable[[Tensor], Tensor],
        execution_horizon=None,
    ) -> Tensor:
        v_t = original_denoise_step_partial(x_t)
        if self.eta != 0.0 and self.step % self.apply_every == 0:
            v_t, loss, gnorm = ogg_velocity(
                x_t, v_t, time, self.dynamics, self.z_t, self.target_delta, self.eta, self.action_dim
            )
            self.losses.append(loss)
            self.grad_norms.append(gnorm)
        self.step += 1
        return v_t


def guided_euler(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    num_steps: int,
    processor: OGGProcessor | None,
) -> Tensor:
    bsize = noise.shape[0]
    dt = -1.0 / float(num_steps)
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.full((bsize,), time, dtype=torch.float32, device=noise.device)
        if processor is None:
            v_t = denoise_fn(x_t, time_tensor)
        else:
            v_t = processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=None,
                inference_delay=None,
                time=time,
                original_denoise_step_partial=lambda x, tt=time_tensor: denoise_fn(x, tt),
            )
        x_t = x_t + dt * v_t
    return x_t
