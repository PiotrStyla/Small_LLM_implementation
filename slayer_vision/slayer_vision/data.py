"""Dataset: zdjęcie → krótkie zdanie po polsku.

Format `captions.jsonl` — jedna linia na próbkę:
    {"image": "lub/ścieżka/bezwzględna.jpg", "text": "To mleko. Termin ważności 12 marca."}
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from slayer_vision.config import IMAGE_SIZE


def load_image_tensor(path: Path) -> torch.Tensor:
    """Wczytuje obraz i normalizuje jak SigLIP: piksel [0,255] → [-1,1]."""
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


class CaptionDataset(Dataset):
    """Pary (obraz, zdanie PL) — tekst nadzorowany w całości, obraz bez etykiet."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        image_root: str = "",
        max_text_tokens: int = 96,
        augment: bool = False,
    ) -> None:
        self.image_root = Path(image_root) if image_root else None
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.augment = augment
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "image" not in record or "text" not in record:
                    raise ValueError(f"{jsonl_path}:{line_number} bez pól image/text")
                self.samples.append(record)
        if not self.samples:
            raise ValueError(f"{jsonl_path} nie zawiera próbek")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, name: str) -> Path:
        path = Path(name)
        return path if path.is_absolute() or self.image_root is None else self.image_root / path

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        pixel_values = load_image_tensor(self._resolve(record["image"]))
        if self.augment:
            from slayer_vision.augment import degrade

            pixel_values = degrade(pixel_values)
        ids = self.tokenizer(
            record["text"],
            truncation=True,
            max_length=self.max_text_tokens,
        )["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids = ids + [eos]
        input_ids = torch.tensor(ids, dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            # Nadzorujemy całe zdanie: LM uczy się generować opis po tokenach
            # obrazu (maska -100 na obraz dokłada model w forwardze).
            "labels": input_ids.clone(),
        }


def collate(batch: list, pad_token_id: int) -> dict:
    """Padding do najdłuższego tekstu w batchu; obrazy składane w stos."""
    max_length = max(item["input_ids"].shape[0] for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        length = item["input_ids"].shape[0]
        pad = max_length - length
        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        attention.append(
            torch.cat(
                [torch.ones(length, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
            )
        )
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention),
    }
