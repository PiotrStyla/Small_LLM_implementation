"""Eksperyment „czy zbliżenie odblokuje cyfry?" — zbiór cropów slotów.

Wniosek z wymiany 10: VLM 224×224 nie czyta cyfr z dokumentów (0/24 slotów),
prawdopodobnie dlatego, że cyfry na pełnym obrazie mają 3–4 px. Test:
dla scen cyfrowych (`synth.DIGIT_KINDS`) wycinamy **zbliżenie wartości slotu**
(bbox z renderera + margines) i uczymy/ewaluujemy na takich cropach. Jeśli
trafność cyfr wystrzeli — potwierdzone: w produkcie trzeba toru detekcja→crop.

Użycie:
    python -m slayer_vision.build_digit_probe --out out/digit-probe --per-kind 80
"""

import argparse
from pathlib import Path

from PIL import Image

from slayer_vision import synth
from slayer_vision.build_dataset import _split

CROP_PAD = 0.25  # margines wokół boksa slotu (w stosunku do rozmiaru boksa)


def crop_slot(image: Image.Image, box: tuple[int, int, int, int], pad: float = CROP_PAD) -> Image.Image:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad) + 8, int(h * pad) + 8
    region = (
        max(0, x0 - px),
        max(0, y0 - py),
        min(image.width, x1 + px),
        min(image.height, y1 + py),
    )
    return image.crop(region).resize((224, 224), Image.LANCZOS)


def build(out_dir: Path, per_kind: int, seed: int) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for kind in synth.DIGIT_KINDS:
        for index in range(per_kind):
            kind_seed = seed * 100_000 + hash(kind) % 10_000 + index
            image, sentence, boxes = synth.render_full(kind, kind_seed)
            slot_box = boxes.get("kwota") or boxes.get("data") or boxes.get("cena")
            if slot_box is None:
                continue  # np. „list" bez kwoty — poza eksperymentem
            crop = crop_slot(image, slot_box)
            name = f"crop__{kind}__{index:04d}.png"
            crop.save(images_dir / name)
            records.append({"image": name, "text": sentence, "scene": f"digit/{kind}"})

    train, eval_ = _split(records)
    for name, part in (("train", train), ("eval", eval_)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for record in part:
                import json

                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Cropy: {len(records)} (train {len(train)}, eval {len(eval_)}) → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zbiór cropów slotów (eksperyment cyfrowy)")
    parser.add_argument("--out", default="out/digit-probe")
    parser.add_argument("--per-kind", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(Path(args.out), args.per_kind, args.seed)


if __name__ == "__main__":
    main()
