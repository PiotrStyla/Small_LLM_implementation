"""Merges caption JSONL files that use different image roots.

Each input keeps its own image root by rewriting relative image paths as
absolute ones (CaptionDataset resolves absolute paths unchanged).

Usage:
    python -m slayer_vision.merge_caption_sets --out out/mixed/train.jsonl \
        out/dataset/train.jsonl out/dataset/images \
        out/coco-objects/train-target-classes.jsonl out/coco-objects/images
"""

import argparse
import json
from pathlib import Path


def merge(out: Path, inputs: list[tuple[Path, Path]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as destination:
        for jsonl_path, image_root in inputs:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                image = Path(record["image"])
                record["image"] = str((image if image.is_absolute() else image_root / image).resolve())
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    print(f"Zapisano {total} próbek: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("pairs", nargs="+", type=Path, help="naprzemiennie: zbior.jsonl katalog_obrazow")
    args = parser.parse_args()
    if len(args.pairs) % 2:
        raise SystemExit("Podaj pary: zbior.jsonl katalog_obrazow")
    merge(args.out, list(zip(args.pairs[::2], args.pairs[1::2])))


if __name__ == "__main__":
    main()
