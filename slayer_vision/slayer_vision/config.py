"""Stałe i konfiguracja treningu SLAYER-Vision-PL."""

from dataclasses import dataclass

# Baza językowa: polski GPT-2 (12×768, ctx 512, vocab 32000), instruction-tuned.
GOLLEM_REPO = "SlayerLab/goLLeM-110M-PL-SFT-merged"

# Zamrożony encoder obrazu: SigLIP-base patch16/224 (hidden 768, 12 warstw).
SIGLIP_REPO = "google/siglip-base-patch16-224"

IMAGE_SIZE = 224
PATCH_SIZE = 16
# Liczba tokenów obrazu: (224/16)^2 = 196 patchy (SigLIP bez tokena CLS).
IMAGE_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2

# GoLLeM ma kontekst 512: 196 tokenów obrazu + tekst + odpowiedź muszą się
# zmieścić. Przy pytaniu ~20 tokenów zostaje ~290 na odpowiedź — wystarczy,
# bo produkt wymusza krótkie jednozdaniowe odpowiedzi.
MAX_TOTAL_TOKENS = 512


@dataclass
class TrainConfig:
    """Parametry treningu projectora + LoRA (vision encoder zamrożony)."""

    data_jsonl: str = "data/captions.jsonl"
    image_root: str = "data/images"
    output_dir: str = "out/slayer-vision"
    steps: int = 1000
    batch_size: int = 4
    max_text_tokens: int = 96
    lr_projector: float = 1e-3
    lr_lora: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 10
    save_every: int = 250
    hf_token: str | None = None
