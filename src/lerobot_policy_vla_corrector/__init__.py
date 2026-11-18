try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError("lerobot is not installed: pip install lerobot==0.7.0") from err

from .configuration_vla_corrector import VLACorrectorConfig
from .modeling_vla_corrector import VLACorrectorPolicy
from .processor_vla_corrector import (
    make_vla_corrector_pre_post_processors,
    make_vla_corrector_pre_post_processors_from_pretrained,
)

__version__ = "0.1.0"

__all__ = [
    "VLACorrectorConfig",
    "VLACorrectorPolicy",
    "make_vla_corrector_pre_post_processors",
    "make_vla_corrector_pre_post_processors_from_pretrained",
]
