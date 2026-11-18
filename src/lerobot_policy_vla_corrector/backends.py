from __future__ import annotations

from contextlib import contextmanager

import torch
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import Tensor

from .corrector import OGGProcessor


class SmolVLABackend:
    name = "smolvla"

    def __init__(self, base: PreTrainedPolicy):
        if base.name != "smolvla":
            raise ValueError(f"the smolvla backend needs a SmolVLA base policy, got {base.name!r}")
        if base.config.rtc_config is not None and base.config.rtc_config.enabled:
            raise ValueError("vla_corrector does not support a base policy with RTC enabled")
        self.base = base
        self.model = base.model

    @property
    def token_dim(self) -> int:
        return int(self.model.vlm_with_expert.vlm.config.text_config.hidden_size)

    @property
    def chunk_size(self) -> int:
        return int(self.base.config.chunk_size)

    @property
    def n_action_steps(self) -> int:
        return int(self.base.config.n_action_steps)

    @property
    def action_dim(self) -> int:
        return int(self.base.config.action_feature.shape[0])

    @torch.no_grad()
    def latents(self, batch: dict[str, Tensor]) -> tuple[Tensor, list[Tensor]]:
        images, masks = self.base.prepare_images(batch)
        tokens = [self.model.vlm_with_expert.embed_image(img) for img in images]
        used = [tok for tok, m in zip(tokens, masks, strict=True) if bool(m[0])]
        if not used:
            used = [tokens[0]]
        return torch.cat(used, dim=1).to(torch.float32), tokens

    @contextmanager
    def reuse_tokens(self, tokens: list[Tensor] | None):
        if tokens is None:
            yield
            return
        vwe = self.model.vlm_with_expert
        it = iter(tokens)
        expected = len(tokens)
        calls = [0]

        def cached(image: Tensor) -> Tensor:
            tok = next(it)
            calls[0] += 1
            if tok.shape[0] != image.shape[0]:
                raise RuntimeError("cached vision tokens do not match the images of this re-plan")
            return tok

        previous = vwe.__dict__.get("embed_image")
        vwe.__dict__["embed_image"] = cached
        try:
            yield
        finally:
            if previous is None:
                vwe.__dict__.pop("embed_image", None)
            else:
                vwe.__dict__["embed_image"] = previous
        if calls[0] != expected:
            raise RuntimeError(f"re-plan consumed {calls[0]} cached token blocks, expected {expected}")

    def predict_chunk(self, batch: dict[str, Tensor], noise: Tensor | None, tokens: list[Tensor] | None) -> Tensor:
        with self.reuse_tokens(tokens):
            return self.base.predict_action_chunk(batch, noise=noise)

    @torch.no_grad()
    def guided_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None,
        processor: OGGProcessor | None,
        tokens: list[Tensor] | None,
    ) -> Tensor:
        base = self.base
        model = self.model
        base.eval()
        batch = base._prepare_batch(batch)
        base._queues = populate_queues(base._queues, batch, exclude_keys=[ACTION])
        for k in batch:
            if k in base._queues and k != ACTION:
                batch[k] = torch.stack(list(base._queues[k]), dim=1)
        images, img_masks = base.prepare_images(batch)
        state = base.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = model.sample_noise((bsize, model.config.chunk_size, model.config.max_action_dim), device)
        else:
            noise = noise.to(device=device, dtype=torch.float32)
        with self.reuse_tokens(tokens):
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
        )

        def denoise(x_t: Tensor, timestep: Tensor) -> Tensor:
            return model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )

        x_t = euler_integrate(
            denoise,
            noise,
            model.config.num_steps,
            rtc_processor=processor,
            rtc_enabled=processor is not None,
        )
        actions = x_t[:, :, : self.action_dim]
        if base.config.adapt_to_pi_aloha:
            actions = base._pi_aloha_encode_actions(actions)
        return actions


BACKENDS = {"smolvla": SmolVLABackend}


def make_backend(name: str, base: PreTrainedPolicy):
    if name not in BACKENDS:
        raise ValueError(f"unknown encoder_backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name](base)
