"""Smoke test SLAYER-Vision-PL na CPU — bez pobierania jakichkolwiek wag.

Buduje model w REALNYCH wymiarach (SigLIP-base 12×768 + GoLLeM 12×768/ctx 512),
ale z losowymi wagami, i przechodzi pełny krok treningu: forward → loss →
backward → optimizer.step(). Dowodzi poprawności okablowania (kształty tensorów,
maskowanie etykiet, przepływ gradientów), nie jakości modelu.

Uruchomienie:  python smoke_test.py
"""

import torch

from slayer_vision.config import IMAGE_TOKENS, IMAGE_SIZE
from slayer_vision.model import SlayerVisionModel
from transformers import GPT2Config, GPT2LMHeadModel, SiglipVisionConfig, SiglipVisionModel

VOCAB_SIZE = 32000
CTX = 512
BATCH = 2
TEXT_TOKENS = 32


def main() -> None:
    vision = SiglipVisionModel(
        SiglipVisionConfig(
            image_size=IMAGE_SIZE,
            patch_size=16,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
        )
    )
    lm = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=CTX,
            n_ctx=CTX,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
    )
    model = SlayerVisionModel(vision=vision, lm=lm, lora_r=16, lora_alpha=32)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"vision hidden: {vision.config.hidden_size}, tokenów obrazu: {model.image_tokens}")
    print(f"parametry łącznie: {total:,}")
    print(f"parametry trenowalne (projector + LoRA): {trainable:,}")

    # Realne budżety: ~86M vision + ~110M LM + ~3-4M projector/LoRA ≈ 200M.
    assert 180_000_000 <= total <= 220_000_000, total
    assert trainable <= 5_000_000, trainable
    assert model.image_tokens == IMAGE_TOKENS == 196

    torch.manual_seed(0)
    pixel_values = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, TEXT_TOKENS))
    attention_mask = torch.ones(BATCH, TEXT_TOKENS, dtype=torch.long)
    labels = input_ids.clone()

    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert out.loss.requires_grad and torch.isfinite(out.loss), out.loss
    print(f"loss po forwardze: {out.loss.item():.4f}")

    out.loss.backward()

    # Gradienty muszą płynąć do projectora i LoRA, nigdy do zamrożonego vision.
    projector_grad = model.projector[0].weight.grad
    assert projector_grad is not None and projector_grad.abs().sum() > 0
    lora_params = [p for n, p in model.lm.named_parameters() if "lora_" in n]
    assert lora_params and all(p.grad is not None for p in lora_params)
    assert all(p.grad is None for p in model.vision.parameters())
    assert all(not p.requires_grad for p in model.vision.parameters())

    before = model.projector[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    optimizer.step()
    assert not torch.equal(before, model.projector[0].weight.detach())

    # Wyrównanie etykiet: pozycje obrazu maskowane, tekst nadzorowany.
    prefix = torch.full((BATCH, model.image_tokens), -100, dtype=labels.dtype)
    full_labels = torch.cat([prefix, labels], dim=1)
    assert full_labels.shape[1] == IMAGE_TOKENS + TEXT_TOKENS <= CTX
    assert (full_labels[:, :IMAGE_TOKENS] == -100).all()

    print("SMOKE OK: forward/backward/step w realnych wymiarach działa.")


if __name__ == "__main__":
    main()
