from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

MODES = ("off", "monitor_only", "truncate", "truncate_ogg")
BACKENDS = ("smolvla",)
LOSSES = ("cosine", "mse", "both")


@PreTrainedConfig.register_subclass("vla_corrector")
@dataclass
class VLACorrectorConfig(PreTrainedConfig):
    base_policy_path: str | None = None
    encoder_backend: str = "smolvla"
    corrector_path: str | None = None
    mode: str = "truncate_ogg"

    k_step: int = 10
    token_dim: int | None = None
    mlp_widths: tuple[int, ...] = (2048, 2048, 2048, 2048)
    action_embed_dim: int = 256
    dropout: float = 0.0
    normalize_latents: bool = False

    window: int = 15
    lambda_on: float = 3.0
    lambda_off: float = 2.0
    patience: int = 5
    cooldown: int = 10
    reset_steps: int = 5
    noise_floor: float = 0.25
    bootstrap_threshold: float = 10.0
    bootstrap_size: int = 15
    ewma_alpha: float = 0.35
    trigger_margin: float = 0.02
    jump_trigger: float = 0.06
    hard_retrigger_margin: float = 0.08
    hard_retrigger_steps: int = 3

    ogg_eta: float = 1.0
    ogg_apply_every: int = 1
    ogg_calls_after_trigger: int = 1

    monitor_every_n_steps: int = 1
    reuse_vision_features: bool = True
    jitter_seed: int = 0

    train_loss: str = "cosine"
    optimizer_lr: float = 3e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.encoder_backend not in BACKENDS:
            raise ValueError(f"encoder_backend must be one of {BACKENDS}, got {self.encoder_backend!r}")
        if self.train_loss not in LOSSES:
            raise ValueError(f"train_loss must be one of {LOSSES}, got {self.train_loss!r}")
        if self.k_step < 1:
            raise ValueError("k_step must be at least 1")
        if self.monitor_every_n_steps < 1 or self.k_step % self.monitor_every_n_steps != 0:
            raise ValueError("monitor_every_n_steps must be at least 1 and divide k_step")
        if len(self.mlp_widths) < 2:
            raise ValueError("mlp_widths needs at least two entries")
        if self.ogg_apply_every < 1:
            raise ValueError("ogg_apply_every must be at least 1")
        if self.ogg_calls_after_trigger < 0:
            raise ValueError("ogg_calls_after_trigger must be 0 or more")
        if self.lambda_off > self.lambda_on:
            raise ValueError("lambda_off must not exceed lambda_on")
        if self.window < 2 or self.bootstrap_size < 1 or self.patience < 1:
            raise ValueError("window must be at least 2, bootstrap_size and patience at least 1")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("vla_corrector needs at least one image feature")
        if self.action_feature is None:
            raise ValueError("vla_corrector needs 'action' in output_features")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def image_observation_delta_indices(self) -> list[int]:
        return [0, self.k_step]

    @property
    def state_observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None
