"""Generuje `photos/REVIEW.html` — podgląd pobranych zdjęć do recenzji.

Wyszukiwarka Commons bywa szumna; człowiek musi szybko przejrzeć, czy zdjęcie
faktycznie pasuje do sceny. Strona pokazuje miniatury pogrupowane po scenie
z tytułem pliku i zapytaniem — do wywalenia nietrafionych wystarczy skasować
plik w `photos/<scene_id>/` i przebudować zbiór.

Użycie:
    python -m slayer_vision.review --photos photos
"""

import argparse
import html
import json
from pathlib import Path


def build(photos_dir: Path) -> None:
    attribution = json.loads(
        (photos_dir / "ATTRIBUTION.json").read_text(encoding="utf-8")
    )
    by_scene: dict[str, list] = {}
    for entry in attribution:
        by_scene.setdefault(entry["scene"], []).append(entry)

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Recenzja zdjęć — SLAYER-Vision</title>",
        "<style>body{font-family:sans-serif;background:#1c1c1c;color:#eee}"
        ".scene{margin:24px 0;border-top:1px solid #444;padding-top:8px}"
        "img{width:180px;height:180px;object-fit:cover;margin:4px}"
        ".muted{color:#999;font-size:12px}</style>",
        f"<h1>Recenzja {len(attribution)} zdjęć ({len(by_scene)} scen)</h1>",
        "<p class='muted'>Nietrafione? Skasuj plik w photos/&lt;scene_id&gt;/ "
        "i przebuduj zbiór (build_dataset).</p>",
    ]
    for scene in sorted(by_scene):
        entries = by_scene[scene]
        parts.append(f"<div class='scene'><h2>{html.escape(scene)}</h2>")
        for entry in entries:
            title = html.escape(entry.get("title", ""))
            query = html.escape(entry.get("query", ""))
            parts.append(
                f"<figure style='display:inline-block;text-align:center'>"
                f"<img loading='lazy' src='{html.escape(entry['file'])}'>"
                f"<figcaption class='muted'>{title}<br>q: {query}</figcaption>"
                f"</figure>"
            )
        parts.append("</div>")

    target = photos_dir / "REVIEW.html"
    target.write_text("\n".join(parts), encoding="utf-8")
    print(f"Zapisano: {target} ({len(attribution)} zdjęć, {len(by_scene)} scen)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generuje stronę recenzji zdjęć")
    parser.add_argument("--photos", default="photos")
    args = parser.parse_args()
    build(Path(args.photos))


if __name__ == "__main__":
    main()
