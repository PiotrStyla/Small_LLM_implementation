"""Trening SLAYER-Vision-PL: projector MLP + LoRA na zamrożonym GoLLeM-110M.

Użycie:
    python -m slayer_vision.train --data-jsonl data/captions.jsonl \
        --image-root data/images --steps 1000 --batch-size 4
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, SiglipVisionModel

from slayer_vision.config import GOLLEM_REPO, SIGLIP_REPO, TrainConfig
from slayer_vision.data import CaptionDataset, collate
from slayer_vision.model import SlayerVisionModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config: TrainConfig) -> tuple[SlayerVisionModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(GOLLEM_REPO, token=config.hf_token)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer GoLLeM nie ma tokena EOS — sprawdź repo modelu.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision = SiglipVisionModel.from_pretrained(SIGLIP_REPO, token=config.hf_token)
    lm = GPT2LMHeadModel.from_pretrained(GOLLEM_REPO, token=config.hf_token)
    model = SlayerVisionModel(
        vision=vision,
        lm=lm,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
    )
    return model, tokenizer


def save_checkpoint(model: SlayerVisionModel, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), output_dir / "projector.pt")
    model.lm.save_pretrained(output_dir / "lora")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    manifest = {
        "base_lm": GOLLEM_REPO,
        "vision_encoder": SIGLIP_REPO,
        "image_tokens": model.image_tokens,
    }
    (output_dir / "slayer_vision.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model(config)
    model.to(device)

    dataset = CaptionDataset(
        config.data_jsonl,
        tokenizer,
        image_root=config.image_root,
        max_text_tokens=config.max_text_tokens,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.projector.parameters(), "lr": config.lr_projector},
            {"params": model.lm.parameters(), "lr": config.lr_lora},
        ],
        weight_decay=config.weight_decay,
    )

    trainable = sum(p.numel() for p in model.trainable_parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametry: {total:,} łącznie, {trainable:,} trenowalnych, urządzenie: {device}")

    model.train()
    model.vision.eval()
    step, running = 0, 0.0
    output_dir = Path(config.output_dir)
    while step < config.steps:
        for batch in loader:
            if step >= config.steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            running += loss.item()
            if step % config.log_every == 0:
                print(f"krok {step}/{config.steps}  loss {running / config.log_every:.4f}")
                running = 0.0
            if step % config.save_every == 0 or step == config.steps:
                save_checkpoint(model, tokenizer, output_dir)
                print(f"Zapisano checkpoint: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening SLAYER-Vision-PL")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--max-text-tokens", type=int, default=TrainConfig.max_text_tokens)
    parser.add_argument("--lr-projector", type=float, default=TrainConfig.lr_projector)
    parser.add_argument("--lr-lora", type=float, default=TrainConfig.lr_lora)
    parser.add_argument("--lora-r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    train(
        TrainConfig(
            data_jsonl=args.data_jsonl,
            image_root=args.image_root,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            max_text_tokens=args.max_text_tokens,
            lr_projector=args.lr_projector,
            lr_lora=args.lr_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
            save_every=args.save_every,
            hf_token=args.hf_token,
        )
    )


if __name__ == "__main__":
    main()
