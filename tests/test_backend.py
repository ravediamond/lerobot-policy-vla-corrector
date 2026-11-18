import types

import pytest
import torch

from lerobot_policy_vla_corrector.backends import SmolVLABackend


class FakeVisionExpert:
    def embed_image(self, image):
        return torch.zeros(image.shape[0], 2, 3)


def make_backend():
    backend = object.__new__(SmolVLABackend)
    backend.model = types.SimpleNamespace(vlm_with_expert=FakeVisionExpert())
    return backend


def test_reuse_tokens_serves_cached_blocks_and_removes_the_patch():
    backend = make_backend()
    vwe = backend.model.vlm_with_expert
    cached = [torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0)]
    with backend.reuse_tokens(cached):
        assert torch.equal(vwe.embed_image(torch.zeros(1, 3, 4, 4)), cached[0])
        assert torch.equal(vwe.embed_image(torch.zeros(1, 3, 4, 4)), cached[1])
    assert "embed_image" not in vwe.__dict__


def test_reuse_tokens_restores_an_existing_override():
    backend = make_backend()
    vwe = backend.model.vlm_with_expert

    def compiled(image):
        return torch.full((image.shape[0], 2, 3), 7.0)

    vwe.__dict__["embed_image"] = compiled
    with backend.reuse_tokens([torch.ones(1, 2, 3)]):
        vwe.embed_image(torch.zeros(1, 3, 4, 4))
    assert vwe.__dict__["embed_image"] is compiled


def test_reuse_tokens_checks_the_block_count():
    backend = make_backend()
    with pytest.raises(RuntimeError):
        with backend.reuse_tokens([torch.ones(1, 2, 3), torch.ones(1, 2, 3)]):
            backend.model.vlm_with_expert.embed_image(torch.zeros(1, 3, 4, 4))
