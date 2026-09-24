"""SLAYER-Vision-PL — ultramały polski VLM na bazie modeli SlayerLab.

Schemat: obraz → zamrożony SigLIP-base → projector MLP → tokeny obrazu
dołączone do tokenów tekstu → goLLeM-110M-PL-SFT (LoRA) → zdanie po polsku.
"""

from slayer_vision.config import TrainConfig, GOLLEM_REPO, SIGLIP_REPO
from slayer_vision.model import SlayerVisionModel

__all__ = ["TrainConfig", "SlayerVisionModel", "GOLLEM_REPO", "SIGLIP_REPO"]
