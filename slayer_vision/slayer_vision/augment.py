"""Degradacje obrazu — odporność na realne warunki zdjęcia.

Śladem augmentacji z PolOCRBench/OCR_engine: jasność, kontrast, rozmycie,
szum, drobny obrót, kwantyzacja (zamiennik artefaktów JPEG). Pracujemy na
tensorze (3, H, W) w zakresie [-1, 1], bez zależności od torchvision.

Użycie treningowe: `CaptionDataset(..., augment=True)`.
Pomiar odporności: `evaluate --degrade` (deterministycznie per próbka).
"""

import math
import random

import torch
import torch.nn.functional as F


def _brightness(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    return (x + rng.uniform(-0.25, 0.25)).clamp(-1, 1)


def _contrast(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    factor = rng.uniform(0.7, 1.3)
    mean = x.mean()
    return ((x - mean) * factor + mean).clamp(-1, 1)


def _blur(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    sigma = rng.uniform(0.6, 1.8)
    coords = torch.arange(5, dtype=x.dtype) - 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_2d = (gauss[:, None] * gauss[None, :]).to(x.dtype)
    kernel_2d = kernel_2d / kernel_2d.sum()
    weight = kernel_2d.expand(3, 1, 5, 5)
    return F.conv2d(x.unsqueeze(0), weight, padding=2, groups=3).squeeze(0)


def _noise(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    # Generator seedowany z [rng] — globalny torch RNG łamie determinizm
    # przy `degrade(..., rng=Random(seed))`.
    gen = torch.Generator().manual_seed(rng.randrange(2**31))
    noise = torch.randn(x.shape, dtype=x.dtype, generator=gen)
    return (x + noise * rng.uniform(0.01, 0.08)).clamp(-1, 1)


def _rotate(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    angle = math.radians(rng.uniform(-6, 6))
    cos, sin = math.cos(angle), math.sin(angle)
    theta = torch.tensor(
        [[[cos, -sin, 0.0], [sin, cos, 0.0]]], dtype=x.dtype
    )
    grid = F.affine_grid(theta, x.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(
        x.unsqueeze(0), grid, align_corners=False, padding_mode="border"
    ).squeeze(0)


def _quantize(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    levels = rng.choice([8, 12, 16])
    return (x * levels).round() / levels


OPS = (_brightness, _contrast, _blur, _noise, _rotate, _quantize)


def degrade(
    pixel_values: torch.Tensor,
    rng: random.Random | None = None,
    max_ops: int = 3,
) -> torch.Tensor:
    """Losowo 1..[max_ops] degradacji z [OPS]; zwraca tensor w [-1, 1]."""
    rng = rng or random.Random()
    x = pixel_values
    for op in rng.sample(OPS, k=rng.randint(1, max_ops)):
        x = op(x, rng)
    return x.clamp(-1, 1)
