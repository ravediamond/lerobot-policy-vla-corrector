from collections import deque
from dataclasses import dataclass

import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor, nn

from lerobot_policy_vla_corrector.corrector import guided_euler

IMG = "observation.images.cam"
STATE_DIM = 4
ACTION_DIM = 3
TOKENS = 5
TOKEN_DIM = 8
CHUNK = 12
N_ACTION_STEPS = 10
NUM_STEPS = 4


@PreTrainedConfig.register_subclass("stub_flow")
@dataclass
class StubFlowConfig(PreTrainedConfig):
    chunk_size: int = CHUNK
    n_action_steps: int = N_ACTION_STEPS
    num_steps: int = NUM_STEPS

    def validate_features(self) -> None:
        return None

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None


class StubFlowPolicy(PreTrainedPolicy):
    config_class = StubFlowConfig
    name = "stub_flow"

    def __init__(self, config: StubFlowConfig, **kwargs):
        super().__init__(config)
        self.config = config
        g = torch.Generator().manual_seed(0)
        self.state_to_action = nn.Linear(STATE_DIM, ACTION_DIM)
        self.image_to_tokens = nn.Linear(3 * 4 * 4, TOKENS * TOKEN_DIM)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        self.reset()

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def velocity(self, x_t: Tensor, timestep: Tensor, target: Tensor) -> Tensor:
        return (x_t - target) * (1.0 + 0.1 * timestep.view(-1, 1, 1))

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        state = batch[OBS_STATE]
        target = self.state_to_action(state)[:, None, :].expand(-1, self.config.chunk_size, -1)
        if noise is None:
            noise = self.sample_noise(target.shape, state.device)
        return guided_euler(lambda x, t: self.velocity(x, t, target), noise, self.config.num_steps, None)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        return self.sample_actions(batch, noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def forward(self, batch):
        raise NotImplementedError


class StubBackend:
    name = "stub"

    def __init__(self, base: StubFlowPolicy):
        self.base = base

    @property
    def token_dim(self) -> int:
        return TOKEN_DIM

    @property
    def chunk_size(self) -> int:
        return self.base.config.chunk_size

    @property
    def n_action_steps(self) -> int:
        return self.base.config.n_action_steps

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @torch.no_grad()
    def latents(self, batch):
        img = batch[IMG]
        if img.ndim == 5:
            img = img[:, -1]
        flat = img.reshape(img.shape[0], -1).to(torch.float32)
        tokens = torch.tanh(self.base.image_to_tokens(flat)).reshape(img.shape[0], TOKENS, TOKEN_DIM)
        return tokens, [tokens]

    def predict_chunk(self, batch, noise, tokens):
        return self.base.predict_action_chunk(batch, noise=noise)

    @torch.no_grad()
    def guided_chunk(self, batch, noise, processor, tokens):
        state = batch[OBS_STATE]
        target = self.base.state_to_action(state)[:, None, :].expand(-1, self.base.config.chunk_size, -1)
        if noise is None:
            noise = self.base.sample_noise(target.shape, state.device)
        return guided_euler(lambda x, t: self.base.velocity(x, t, target), noise, self.base.config.num_steps, processor)


def stub_features():
    return (
        {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            IMG: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )


def make_stub():
    inputs, outputs = stub_features()
    cfg = StubFlowConfig(input_features=inputs, output_features=outputs, device="cpu")
    return StubFlowPolicy(cfg)


def observation(step: int, b: int = 1, seed: int = 0) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed * 100003 + step)
    return {
        OBS_STATE: torch.randn(b, STATE_DIM, generator=g),
        IMG: torch.rand(b, 3, 4, 4, generator=g),
    }
