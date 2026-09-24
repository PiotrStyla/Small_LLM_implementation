"""Buduje dataset SLAYER-Vision-PL: obrazy + `train.jsonl`/`eval.jsonl`.

Dwa źródeł próbek:
1. syntetyka (`synth`): renderowane dokumenty/panele z dokładnym GT;
2. zdjęcia: `photos/<scene_id>/*.jpg` zrobione do katalogu `scenes`.

Podział train/eval: co piąta próbka grupy (sceny/typu syntetyki) idzie do
eval — ten sam obiekt, inne obrazy, czyli pomiar generalizacji na nowe
ujęcie, nie na nową kategorię.

Użycie:
    python -m slayer_vision.build_dataset --out out/dataset \
        --per-kind 40 --photos photos
"""

import argparse
import json
import random
from pathlib import Path

from slayer_vision import synth
from slayer_vision.scenes import SCENES


def _split(records: list, eval_every: int = 5) -> tuple[list, list]:
    """Podział per grupa (scena/typ syntetyki): co [eval_every]-ta próbka →
    eval, a małe grupy (np. 4 zdjęcia) oddają ostatnią próbkę — inaczej
    nie miałyby wcale reprezentacji w eval."""
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(record["scene"], []).append(record)
    train, eval_ = [], []
    for items in groups.values():
        held = {index for index in range(len(items)) if (index + 1) % eval_every == 0}
        if not held and len(items) >= 2:
            held = {len(items) - 1}
        for index, item in enumerate(items):
            (eval_ if index in held else train).append(item)
    return train, eval_


def build(out_dir: Path, per_kind: int, photos_dir: Path | None, seed: int) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict] = []

    # 1) Syntetyka z pełnym GT.
    for kind in synth.RENDERERS:
        for index in range(per_kind):
            kind_seed = rng.randrange(2**31)
            image, sentence = synth.render(kind, kind_seed)
            name = f"synth__{kind}__{index:04d}.png"
            image.save(images_dir / name)
            records.append(
                {"image": name, "text": sentence, "scene": f"synth/{kind}"}
            )

    # 2) Zdjęcia użytkownika do katalogu scen.
    missing = []
    photos_used = 0
    if photos_dir and photos_dir.exists():
        for scene in SCENES:
            scene_dir = photos_dir / scene.id
            scene_dir.mkdir(parents=True, exist_ok=True)  # katalog na zdjęcia
            photos = sorted(scene_dir.glob("*.jpg"))
            if not photos:
                missing.append(scene.id)
                continue
            for photo in photos:
                name = f"photo__{scene.id.replace('/', '__')}__{photo.name}"
                (images_dir / name).write_bytes(photo.read_bytes())
                records.append(
                    {"image": name, "text": scene.sentence, "scene": scene.id}
                )
                photos_used += 1
    else:
        missing = [scene.id for scene in SCENES]

    if not records:
        raise SystemExit("Brak próbek — wygeneruj syntetykę albo dodaj zdjęcia.")

    train, eval_ = _split(records)
    for name, part in (("train", train), ("eval", eval_)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for record in part:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (out_dir / "photos_missing.txt").write_text(
        "\n".join(missing) + ("\n" if missing else ""), encoding="utf-8"
    )

    kinds = len(synth.RENDERERS)
    print(f"Próbki: {len(records)} (train {len(train)}, eval {len(eval_)})")
    print(f"Syntetyka: {kinds} typów × {per_kind} = {kinds * per_kind}")
    print(f"Zdjęcia: {photos_used} plików dla {len(SCENES) - len(missing)}/{len(SCENES)} scen")
    print(f"Sceny bez zdjęć: {len(missing)} (lista: {out_dir / 'photos_missing.txt'})")
    print(f"Wyjście: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Buduje dataset SLAYER-Vision-PL")
    parser.add_argument("--out", default="out/dataset")
    parser.add_argument("--per-kind", type=int, default=40, help="obrazów syntetycznych na typ")
    parser.add_argument("--photos", default="", help="katalog zdjęć photos/<scene_id>/*.jpg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(
        Path(args.out),
        args.per_kind,
        Path(args.photos) if args.photos else None,
        args.seed,
    )


if __name__ == "__main__":
    main()
