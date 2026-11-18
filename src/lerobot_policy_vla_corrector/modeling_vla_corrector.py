from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from safetensors.torch import load_file, save_file
from torch import Tensor

from .backends import make_backend
from .configuration_vla_corrector import VLACorrectorConfig
from .corrector import LatentDynamicsMLP, Monitor, OGGProcessor, ThresholdConfig, dynamics_loss

CORRECTOR_PREFIX = "corrector."
WEIGHTS_FILE = "model.safetensors"
TELEMETRY_KEYS = (
    "step",
    "score",
    "threshold",
    "release",
    "triggered",
    "truncated",
    "replanned",
    "ogg_calls",
    "ogg_loss",
    "ogg_grad_norm",
    "monitor_ms",
    "step_ms",
)


def load_base_policy(config: VLACorrectorConfig) -> PreTrainedPolicy:
    if config.base_policy_path is None:
        raise ValueError("vla_corrector needs base_policy_path (a pretrained flow-matching policy)")
    base_cfg = PreTrainedConfig.from_pretrained(config.base_policy_path)
    base_cfg.device = config.device
    if config.dtype is not None:
        base_cfg.dtype = config.dtype
    policy_cls = get_policy_class(base_cfg.type)
    base = policy_cls.from_pretrained(config.base_policy_path, config=base_cfg)
    if config.dtype is not None:
        base.to(config.dtype)
    return base


class VLACorrectorPolicy(PreTrainedPolicy):
    config_class = VLACorrectorConfig
    name = "vla_corrector"

    def __init__(self, config: VLACorrectorConfig, base: PreTrainedPolicy | None = None, backend=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.base = base if base is not None else load_base_policy(config)
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.backend = backend if backend is not None else make_backend(config.encoder_backend, self.base)
        if not config.input_features:
            config.input_features = dict(self.base.config.input_features)
        if not config.output_features:
            config.output_features = dict(self.base.config.output_features)
        config.validate_features()
        token_dim = config.token_dim if config.token_dim is not None else int(self.backend.token_dim)
        config.token_dim = token_dim
        self.corrector = LatentDynamicsMLP(
            token_dim=token_dim,
            action_dim=int(config.action_feature.shape[0]),
            widths=tuple(config.mlp_widths),
            action_embed_dim=config.action_embed_dim,
            dropout=config.dropout,
            normalize_latents=config.normalize_latents,
        )
        self.monitor = Monitor(
            self.corrector,
            k_step=config.k_step,
            thresholds=ThresholdConfig(
                window=config.window,
                noise_floor=config.noise_floor,
                bootstrap_threshold=config.bootstrap_threshold,
                bootstrap_size=config.bootstrap_size,
                ewma_alpha=config.ewma_alpha,
                lambda_on=config.lambda_on,
                lambda_off=config.lambda_off,
                trigger_margin=config.trigger_margin,
                jump_trigger=config.jump_trigger,
                patience=config.patience,
                reset_steps=config.reset_steps,
                cooldown=config.cooldown,
                hard_retrigger_margin=config.hard_retrigger_margin,
                hard_retrigger_steps=config.hard_retrigger_steps,
            ),
            every_n=config.monitor_every_n_steps,
            jitter_seed=config.jitter_seed,
        )
        if config.corrector_path is not None:
            self.load_corrector(config.corrector_path)
        self.telemetry: list[dict] = []
        self.last_info: dict = {}
        self.reset()

    def get_optim_params(self):
        return self.corrector.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def reset(self):
        self.base.reset()
        self.monitor.reset()
        self._queue: deque[Tensor] = deque()
        self._step = 0
        self._pending_ogg = 0
        self._ogg_payload: tuple[Tensor, Tensor] | None = None
        self._ogg_calls = 0
        self.last_info = {}

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config._save_pretrained(save_directory)
        state = {
            k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if k.startswith(CORRECTOR_PREFIX)
        }
        save_file(state, str(save_directory / WEIGHTS_FILE), metadata={"format": "pt"})

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        state = load_file(model_file, device="cpu")
        corrector_state = {k[len(CORRECTOR_PREFIX) :]: v for k, v in state.items() if k.startswith(CORRECTOR_PREFIX)}
        missing, unexpected = model.corrector.load_state_dict(corrector_state, strict=False)
        if missing or unexpected or not corrector_state:
            raise RuntimeError(
                f"corrector weights do not match: missing {sorted(missing)}, unexpected {sorted(unexpected)}"
            )
        return model

    def load_corrector(self, path: str | Path) -> None:
        path = Path(path)
        file = path / WEIGHTS_FILE if path.is_dir() else path
        self._load_as_safetensor(self, str(file), self.config.device, True)
        self.corrector.to(next(self.base.parameters()).device)

    def _can_monitor(self, batch: dict[str, Tensor]) -> bool:
        return self.config.mode != "off" and self.monitor.active(self._step)

    def _state_batch_size(self, batch: dict[str, Tensor]) -> int:
        for v in batch.values():
            if isinstance(v, Tensor) and v.ndim >= 1:
                return int(v.shape[0])
        return 1

    def _plan(self, batch: dict[str, Tensor], noise: Tensor | None, tokens) -> tuple[Tensor, dict]:
        info = {"ogg_loss": float("nan"), "ogg_grad_norm": float("nan")}
        if self.config.mode == "truncate_ogg" and self._pending_ogg > 0 and self._ogg_payload is not None:
            z_t, target = self._ogg_payload
            processor = OGGProcessor(
                self.corrector,
                z_t,
                target,
                eta=self.config.ogg_eta,
                apply_every=self.config.ogg_apply_every,
                action_dim=self.action_dim,
            )
            chunk = self.backend.guided_chunk(batch, noise, processor, tokens)
            self._pending_ogg -= 1
            self._ogg_calls += 1
            if processor.losses:
                info["ogg_loss"] = processor.losses[0]
                info["ogg_grad_norm"] = processor.grad_norms[0]
            return chunk, info
        return self.backend.predict_chunk(batch, noise, tokens), info

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        if self.config.mode == "off":
            return self.base.predict_action_chunk(batch, noise=noise)
        chunk, _ = self._plan(batch, noise, None)
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        if self.config.mode == "off":
            return self.base.select_action(batch, noise=noise)
        t_start = time.perf_counter()
        step = self._step
        info = {k: float("nan") for k in TELEMETRY_KEYS}
        info.update(step=step, triggered=False, truncated=0, replanned=False, ogg_calls=self._ogg_calls)
        z_t = None
        tokens = None
        if self._can_monitor(batch):
            if self._state_batch_size(batch) != 1:
                raise ValueError("the monitor runs with batch size 1; use mode=off for batched inference")
            t0 = time.perf_counter()
            z_batch, tokens = self.backend.latents(batch)
            z_t = z_batch[0]
            result = self.monitor.observe(step, z_t)
            info.update(score=result.score, threshold=result.threshold, release=result.release)
            if result.triggered:
                info["triggered"] = True
                if self.config.mode in ("truncate", "truncate_ogg"):
                    info["truncated"] = len(self._queue)
                    self._queue.clear()
                    if self.config.mode == "truncate_ogg":
                        self._pending_ogg = self.config.ogg_calls_after_trigger
                        self._ogg_payload = (z_t.unsqueeze(0), (result.delta_pred - result.delta_real).unsqueeze(0))
            info["monitor_ms"] = (time.perf_counter() - t0) * 1000.0
            if not self.config.reuse_vision_features:
                tokens = None
        if len(self._queue) == 0:
            chunk, plan_info = self._plan(batch, noise, tokens)
            info.update(plan_info)
            info["replanned"] = True
            self._queue.extend(chunk.transpose(0, 1)[: self.backend.n_action_steps])
        action = self._queue.popleft()
        if z_t is not None:
            self.monitor.predict(step, z_t, action[0, : self.action_dim])
        info["ogg_calls"] = self._ogg_calls
        info["step_ms"] = (time.perf_counter() - t_start) * 1000.0
        self.last_info = info
        self.telemetry.append(info)
        logging.debug("vla_corrector step %s", info)
        self._step += 1
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        frames_now = {}
        frames_next = {}
        pad = None
        for key in self.base.config.image_features:
            if key not in batch:
                continue
            img = batch[key]
            if img.ndim != 5 or img.shape[1] != 2:
                raise ValueError(f"{key} must be (B, 2, C, H, W): frames t and t+k_step, got {tuple(img.shape)}")
            frames_now[key] = img[:, 0]
            frames_next[key] = img[:, 1]
            is_pad = batch.get(f"{key}_is_pad")
            if is_pad is not None:
                pad = is_pad[:, 1] if pad is None else (pad | is_pad[:, 1])
        if not frames_now:
            raise ValueError("no image feature of the base policy found in the batch")
        with torch.no_grad():
            z_t, _ = self.backend.latents(frames_now)
            z_next, _ = self.backend.latents(frames_next)
        actions = batch[ACTION]
        a_t = actions[:, 0] if actions.ndim == 3 else actions
        a_t = a_t[:, : self.action_dim].to(z_t)
        pred = self.corrector(z_t, a_t)
        target = z_next - z_t
        if pad is not None:
            keep = ~pad.bool()
            if int(keep.sum()) == 0:
                return pred.sum() * 0.0, {"loss_cosine": float("nan"), "loss_mse": float("nan"), "n_valid": 0}
            pred = pred[keep]
            target = target[keep]
        loss, info = dynamics_loss(pred, target, self.config.train_loss)
        info["n_valid"] = int(pred.shape[0])
        return loss, info
