"""Ewaluacja SLAYER-Vision-PL: generowanie zdań do zdjęć z eval.jsonl.

Dekodowanie greedy jest ręczne (pętla po tokenach, bez `generate()`), bo
wejście wielomodalne (tokeny obrazu przed tekstem) wymaga pełnej kontroli nad
`inputs_embeds`. Metryki: trafienie dokładne po normalizacji i F1 tokenów
białych znaków — pomiar „czy model mówi właściwe zdanie dla sceny".

Format treningowy nie ma tokena BOS: model uczy się P(pierwszy token | obraz),
więc generacja startuje z samych tokenów obrazu.

Użycie:
    python -m slayer_vision.evaluate --checkpoint out/run-150 \
        --data-jsonl out/dataset/eval.jsonl --image-root out/dataset/images
"""

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, GPT2LMHeadModel, SiglipVisionModel

from slayer_vision.config import GOLLEM_REPO, SIGLIP_REPO
from slayer_vision.data import CaptionDataset, load_image_tensor
from slayer_vision.model import SlayerVisionModel


def load_checkpoint(
    checkpoint: Path, hf_token: str | None = None
) -> tuple[SlayerVisionModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision = SiglipVisionModel.from_pretrained(SIGLIP_REPO, token=hf_token)
    base_lm = GPT2LMHeadModel.from_pretrained(GOLLEM_REPO, token=hf_token)
    lm = PeftModel.from_pretrained(base_lm, checkpoint / "lora")
    model = SlayerVisionModel(vision=vision, lm=lm)  # lm już z adapterami
    model.projector.load_state_dict(
        torch.load(checkpoint / "projector.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_sentence(
    model: SlayerVisionModel,
    tokenizer: AutoTokenizer,
    pixel_values: torch.Tensor,
    max_new_tokens: int = 24,
) -> str:
    """Greedy: same tokeny obrazu → kolejne tokeny aż do EOS/limitu."""
    device = pixel_values.device
    image_hidden = model.vision(
        pixel_values=pixel_values.unsqueeze(0)
    ).last_hidden_state
    image_embeds = model.projector(image_hidden)

    ids = torch.empty((1, 0), dtype=torch.long, device=device)
    generated: list[int] = []
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        text_embeds = model.lm.get_input_embeddings()(ids) if ids.shape[1] else None
        inputs_embeds = (
            image_embeds
            if text_embeds is None
            else torch.cat([image_embeds, text_embeds], dim=1)
        )
        attention = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )
        logits = model.lm(
            inputs_embeds=inputs_embeds, attention_mask=attention
        ).logits
        next_id = int(logits[0, -1].argmax())
        if eos is not None and next_id == eos:
            break
        generated.append(next_id)
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
        )
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sąćęłńóśźż]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_f1(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference).split()
    hyp = _normalize(hypothesis).split()
    if not ref or not hyp:
        return 1.0 if ref == hyp else 0.0
    common = 0
    ref_counts: dict[str, int] = {}
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in hyp:
        if ref_counts.get(token, 0) > 0:
            common += 1
            ref_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(hyp)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def evaluate(
    checkpoint: Path,
    jsonl_path: str,
    image_root: str,
    max_samples: int | None,
    dump: Path | None,
    hf_token: str | None,
    degrade_inputs: bool = False,
) -> None:
    model, tokenizer = load_checkpoint(checkpoint, hf_token)
    dataset = CaptionDataset(jsonl_path, tokenizer, image_root=image_root)

    rows = []
    for index, record in enumerate(dataset.samples):
        if max_samples is not None and index >= max_samples:
            break
        pixel = load_image_tensor(dataset._resolve(record["image"]))
        if degrade_inputs:
            import random as _random

            from slayer_vision.augment import degrade

            # Deterministycznie per próbka — pomiar degradacji jest powtarzalny.
            pixel = degrade(pixel, rng=_random.Random(index))
        hypothesis = generate_sentence(model, tokenizer, pixel)
        reference = record["text"]
        rows.append(
            {
                "image": record["image"],
                "scene": record.get("scene", ""),
                "reference": reference,
                "hypothesis": hypothesis,
                "exact": _normalize(reference) == _normalize(hypothesis),
                "token_f1": round(_token_f1(reference, hypothesis), 4),
            }
        )

    exact = sum(row["exact"] for row in rows)
    f1 = sum(row["token_f1"] for row in rows) / len(rows)
    print(f"Próbek ewaluowanych: {len(rows)}")
    print(f"Trafienia dokładne: {exact}/{len(rows)} ({100 * exact / len(rows):.1f}%)")
    print(f"Średnie F1 tokenów: {f1:.3f}")
    for row in rows[:10]:
        print(f"  [{row['scene']}] ref: {row['reference']}")
        print(f"       hyp: {row['hypothesis']}  (f1 {row['token_f1']:.2f})")

    if dump is not None:
        with open(dump, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Predykcje zapisane: {dump}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja SLAYER-Vision-PL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dump", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--degrade",
        action="store_true",
        help="pomiar odporności: degraduj obrazy wejściowe (deterministycznie)",
    )
    args = parser.parse_args()
    evaluate(
        Path(args.checkpoint),
        args.data_jsonl,
        args.image_root,
        args.max_samples,
        Path(args.dump) if args.dump else None,
        args.hf_token,
        degrade_inputs=args.degrade,
    )


if __name__ == "__main__":
    main()
