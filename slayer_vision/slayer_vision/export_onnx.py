"""Eksport checkpointu SLAYER-Vision-PL do ONNX dla aplikacji mobilnej.

flutter_gemma uruchamia gotowe rodziny modeli (Gemma/Qwen/…) — naszej własnej
architektury (SigLIP + projector + goLLeM) nie obejmie. Dlatego eksportujemy
DWA grafy ONNX, które aplikacja składa sama:

1. `vision_projector.onnx`: pixel_values (1,3,224,224) → embeddy obrazu
   (1,196,768) — zamrożony SigLIP + wytrenowany projekt; batch stały = 1,
   bo aplikacja przetwarza jedno zdjęcie na zapytanie;
2. `lm_embeds.onnx`: inputs_embeds (B,S,768) + attention_mask → logits
   (B,S,32000) — scalony (merge LoRA) goLLeM-110M-PL-SFT; **S dynamiczne**,
   bo dekodowanie greedy wydłuża sekwencję co krok.

Format treningowy nie ma BOS ani promptu: generacja startuje z samych tokenów
obrazu i dekoduje zdanie greedy. Dlatego aplikacja NIE potrzebuje tokenizera
do kodowania — tylko mapy id → tekst (`tokens_decoded.json`), którą też tu
zapisujemy.

Wybór eksporterów (wnioski z boju):
- vision: `torch.onnx.export` dynamo=True (legacy pada na SigLIP-ie:
  `invalid unordered_map<K, T> key`); batch zaklepany na 1, bo dynamo piecze
  kształty Reshape mimo `dynamic_axes`;
- lm: dynamo=False (legacy) — GPT-2 eksportuje się nim poprawnie z dynamiczną
  sekwencją; opset 18 (niższy wersjonuje węzły Split niepoprawnie).

Weryfikacja: test parzystości PyTorch ↔ ONNX, w tym LM przy sekwencji != niż
eksportowa (dowód dynamiki).

Użycie:
    python -m slayer_vision.export_onnx --checkpoint out/run-final \
        --out out/export-slayer-vision
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel, SiglipVisionModel

from slayer_vision.config import GOLLEM_REPO, IMAGE_SIZE, IMAGE_TOKENS, SIGLIP_REPO
from slayer_vision.evaluate import load_checkpoint


class VisionProjector(nn.Module):
    """pixel_values (1,3,224,224) → tokeny obrazu (1, 196, 768)."""

    def __init__(self, vision: SiglipVisionModel, projector: nn.Module) -> None:
        super().__init__()
        self.vision = vision
        self.projector = projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.vision(pixel_values=pixel_values).last_hidden_state
        return self.projector(hidden)


class LmEmbeds(nn.Module):
    """inputs_embeds + maska → logits (bez cache — MVP, S ≤ ~30 kroków)."""

    def __init__(self, lm: GPT2LMHeadModel) -> None:
        super().__init__()
        self.lm = lm

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.lm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).logits


class EmbedTokens(nn.Module):
    """ids (1,S) int64 → embeddy tokenu (1,S,768).

    Bez tego grafu aplikacja nie potrafi dołożyć wektora kolejnego tokenu
    w dekodowaniu greedy (graf LM je `inputs_embeds`, nie `ids`).
    """

    def __init__(self, lm: GPT2LMHeadModel) -> None:
        super().__init__()
        self.table = lm.get_input_embeddings()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table(ids)


def export(checkpoint: Path, out_dir: Path, hf_token: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_checkpoint(checkpoint, hf_token)
    # Scal LoRA w bazę — na urządzeniu jeden graf, bez adapterów.
    model.lm = model.lm.merge_and_unload()
    model.eval()

    vision_wrapper = VisionProjector(model.vision, model.projector).eval()
    lm_wrapper = LmEmbeds(model.lm).eval()

    dummy_pixels = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dummy_embeds = torch.randn(1, 8, model.lm.config.n_embd)
    dummy_mask = torch.ones(1, 8, dtype=torch.long)

    vision_path = out_dir / "vision_projector.onnx"
    lm_path = out_dir / "lm_embeds.onnx"
    embed_path = out_dir / "embed_tokens.onnx"

    torch.onnx.export(
        EmbedTokens(model.lm).eval(),
        (torch.zeros(1, 2, dtype=torch.long),),
        str(embed_path),
        input_names=["ids"],
        output_names=["embeds"],
        dynamic_axes={"ids": {0: "batch", 1: "seq"}, "embeds": {0: "batch", 1: "seq"}},
        opset_version=18,
    )

    torch.onnx.export(
        vision_wrapper,
        (dummy_pixels,),
        str(vision_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        # Batch nominalnie dynamiczny, ale graf piecze kształty Reshape przy
        # batch>1 (dynamo) — aplikacja używa batch=1 i tak traktujemy graf.
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=18,
    )
    torch.onnx.export(
        lm_wrapper,
        (dummy_embeds, dummy_mask),
        str(lm_path),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=18,
    )

    # --- test parzystości PyTorch ↔ ONNX ---------------------------------
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    pixels = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    # Sekwencja 30 != 8 z eksportu — dowód, że oś `seq` jest dynamiczna.
    ids = torch.randint(0, model.lm.config.vocab_size, (1, 30))
    mask = torch.ones(1, 30, dtype=torch.long)
    with torch.no_grad():
        pt_image = vision_wrapper(pixels)
        embeds = model.lm.get_input_embeddings()(ids)
        pt_logits = lm_wrapper(embeds, mask)

    vision_sess = ort.InferenceSession(str(vision_path), providers=["CPUExecutionProvider"])
    lm_sess = ort.InferenceSession(str(lm_path), providers=["CPUExecutionProvider"])
    onnx_image = vision_sess.run(None, {"pixel_values": pixels.numpy()})[0]
    onnx_logits = lm_sess.run(
        None,
        {"inputs_embeds": embeds.numpy(), "attention_mask": mask.numpy()},
    )[0]

    image_diff = float(abs(pt_image.numpy() - onnx_image).max())
    logits_diff = float(abs(pt_logits.numpy() - onnx_logits).max())
    embed_sess = ort.InferenceSession(str(embed_path), providers=["CPUExecutionProvider"])
    onnx_embeds = embed_sess.run(None, {"ids": ids.numpy()})[0]
    embed_diff = float(abs(embeds.numpy() - onnx_embeds).max())
    print(f"parzystość obrazu: max|Δ| = {image_diff:.2e}")
    print(f"parzystość logitów (S=30 vs eksport S=8): max|Δ| = {logits_diff:.2e}")
    print(f"parzystość embeddów: max|Δ| = {embed_diff:.2e}")

    # --- kryterium produktowe: identyczne zdanie z dekodowania greedy ----
    # Różnice liczbowe rządu 1e-3 to szum fp32 przekształceń grafu; liczy się,
    # czy argmax po drodze wybiera te same tokeny.
    def greedy_pt(image_embeds: torch.Tensor, max_new: int = 24) -> list[int]:
        emb_table = model.lm.get_input_embeddings()
        ids: list[int] = []
        for _ in range(max_new):
            prefix = torch.tensor([ids], dtype=torch.long)
            text = emb_table(prefix) if ids else torch.empty(1, 0, image_embeds.shape[2])
            logits = lm_wrapper(torch.cat([image_embeds, text], dim=1), torch.ones(1, image_embeds.shape[1] + len(ids), dtype=torch.long))
            next_id = int(logits[0, -1].argmax())
            if next_id == tokenizer.eos_token_id:
                break
            ids.append(next_id)
        return ids

    def greedy_onnx(image_embeds: np.ndarray, max_new: int = 24) -> list[int]:
        ids: list[int] = []
        for _ in range(max_new):
            if ids:
                text = embed_sess.run(
                    None, {"ids": np.array([ids], dtype=np.int64)}
                )[0]
            else:
                text = np.zeros((1, 0, model.lm.config.n_embd), dtype=np.float32)
            embeds = np.concatenate([image_embeds, text], axis=1)
            mask = np.ones((1, embeds.shape[1]), dtype=np.int64)
            logits = lm_sess.run(None, {"inputs_embeds": embeds, "attention_mask": mask})[0]
            next_id = int(logits[0, -1].argmax())
            if next_id == tokenizer.eos_token_id:
                break
            ids.append(next_id)
        return ids

    with torch.no_grad():
        pt_ids = greedy_pt(pt_image)
    onnx_ids = greedy_onnx(onnx_image)
    same = pt_ids == onnx_ids
    text = tokenizer.decode(pt_ids)
    print(f"dekodowanie greedy: {'ZGODNE' if same else 'NIEZGODNE'} — {text!r}")
    if not same:
        raise SystemExit(f"NIEZGODNE DEKODOWANIE: pt={pt_ids} onnx={onnx_ids}")
    if image_diff > 5e-3 or logits_diff > 5e-3 or embed_diff > 5e-3:
        raise SystemExit("Różnice liczbowe ponad progiem fp32 — sprawdź graf")

    # --- mapa id → tekst (aplikacja tylko DEKODUJE, nie koduje) ---------
    vocab_size = model.lm.config.vocab_size
    tokens = [tokenizer.decode([i]) for i in range(vocab_size)]
    (out_dir / "tokens_decoded.json").write_text(
        json.dumps(tokens, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "format": "slayer-vision-onnx/1",
        "base_lm": GOLLEM_REPO,
        "vision_encoder": SIGLIP_REPO,
        "checkpoint": str(checkpoint),
        "image_size": IMAGE_SIZE,
        "image_tokens": IMAGE_TOKENS,
        "lm_hidden": model.lm.config.n_embd,
        "vocab_size": vocab_size,
        "files": {
            "vision_projector": vision_path.name,
            "lm": lm_path.name,
            "embed_tokens": embed_path.name,
            "tokens": "tokens_decoded.json",
        },
        "parity": {"image": image_diff, "logits": logits_diff},
        "usage": (
            "image_embeds = vision(pixel_values); potem greedy: "
            "logits = lm(inputs_embeds=[image_embeds, emb(ids)], mask=1) "
            "i argmax ostatniego kroku; tekst = konkatenacja tokens_decoded[id]"
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total = vision_path.stat().st_size + lm_path.stat().st_size
    print(f"Eksport OK: {out_dir} ({total / 1e6:.1f} MB grafów, vocab {vocab_size})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksport SLAYER-Vision-PL do ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="out/export-slayer-vision")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    export(Path(args.checkpoint), Path(args.out), args.hf_token)


if __name__ == "__main__":
    main()
