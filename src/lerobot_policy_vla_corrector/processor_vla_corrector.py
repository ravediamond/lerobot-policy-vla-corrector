from typing import Any

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_vla_corrector import VLACorrectorConfig


def _with_device(overrides: dict[str, Any] | None, device: str | None) -> dict[str, Any]:
    out = dict(overrides or {})
    if device is not None and "device_processor" not in out:
        out["device_processor"] = {"device": device}
    return out


def base_processors(
    config: VLACorrectorConfig,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    revision: str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    base_cfg = PreTrainedConfig.from_pretrained(config.base_policy_path, revision=revision)
    base_cfg.device = config.device
    return make_pre_post_processors(
        base_cfg,
        pretrained_path=config.base_policy_path,
        pretrained_revision=revision,
        preprocessor_overrides=_with_device(preprocessor_overrides, config.device),
        postprocessor_overrides=postprocessor_overrides,
    )


def make_vla_corrector_pre_post_processors(
    config: VLACorrectorConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if config.base_policy_path is None:
        return make_default_pre_post_processors(config, dataset_stats)
    return base_processors(config)


def make_vla_corrector_pre_post_processors_from_pretrained(
    config: VLACorrectorConfig,
    pretrained_path,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str | None = None,
    postprocessor_config_filename: str | None = None,
):
    if config.base_policy_path is None:
        return make_default_pre_post_processors(config, dataset_stats)
    return base_processors(config, preprocessor_overrides, postprocessor_overrides)
