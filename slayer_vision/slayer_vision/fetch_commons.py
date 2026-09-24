"""Pobiera zdjęcia scen z Wikimedia Commons (otwarte licencje) z atrybucją.

Użytkownik nie ma własnych zdjęć — sceny fotograficzne uzupełniamy obrazami
z Commons. Przeszukujemy zapytaniem = nazwa sceny (z zapasowym wariantem z id),
filtrujemy licencje (CC0/PD/CC BY/CC BY-SA — wszystkie z atrybucją albo bez),
zapisujemy miniatury (512 px) do `photos/<scene_id>/` i pełną atrybucję
do `photos/ATTRIBUTION.json` (autor, licencja, URL strony pliku).

Każdy plik ma prefiks `commons__` — łatwo odróżnić od zdjęć własnych i łatwo
usunąć, jeśli recenzja uzna je za niepasujące.

Użycie:
    python -m slayer_vision.fetch_commons --photos photos --per-scene 4
"""

import argparse
import json
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

from slayer_vision.scenes import SCENES

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SlayerVisionPL-dataset/0.1 (open accessibility research)"

# Dozwolone licencje (porównanie po znormalizowanym skrócie).
ALLOWED_LICENSES = {
    "cc0", "cc0 1.0", "public domain", "pd", "cc by 1.0", "cc by 2.0",
    "cc by 2.5", "cc by 3.0", "cc by 4.0", "cc by-sa 1.0", "cc by-sa 2.0",
    "cc by-sa 2.5", "cc by-sa 3.0", "cc by-sa 4.0", "cc by-sa", "cc by",
}


def _api(params: dict) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # Commons throttla (429) przy zbyt gęstych zapytaniach — backoff zamiast
    # padania całego pobrania.
    for attempt in range(5):
        try:
            time.sleep(0.5)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == 4:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def _queries(name: str, scene_id: str) -> list[str]:
    fallback = scene_id.split("/")[-1].replace("-", " ")
    return [name, fallback] if fallback != name else [name]


def _license_ok(extmeta: dict) -> str | None:
    short = extmeta.get("LicenseShortName", {}).get("value", "")
    normalized = short.strip().lower()
    if normalized in ALLOWED_LICENSES:
        return short
    return None


def fetch(per_scene: int, photos_dir: Path, only: str | None = None) -> None:
    photos_dir.mkdir(parents=True, exist_ok=True)
    attribution_path = photos_dir / "ATTRIBUTION.json"
    attribution = (
        json.loads(attribution_path.read_text(encoding="utf-8"))
        if attribution_path.exists()
        else []
    )
    seen = {entry["file"] for entry in attribution}

    for scene in SCENES:
        if only and only not in scene.id:
            continue
        scene_dir = photos_dir / scene.id
        scene_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(scene_dir.glob("commons__*")))
        if existing >= per_scene:
            continue
        want = per_scene - existing
        got = 0
        for query in _queries(scene.name, scene.id):
            if got >= want:
                break
            data = _api({
                "action": "query",
                "generator": "search",
                "gsrnamespace": "6",
                "gsrlimit": str(max(want * 3, 10)),
                "gsrsearch": f"{query} filetype:bitmap",
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime",
                "iiurlwidth": "512",
            })
            for page in data.get("query", {}).get("pages", []):
                if got >= want:
                    break
                infos = page.get("imageinfo", [])
                if not infos:
                    continue
                info = infos[0]
                mime = info.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue
                extmeta = info.get("extmetadata", {})
                license_name = _license_ok(extmeta)
                if license_name is None:
                    continue
                thumb = info.get("thumburl") or info.get("url")
                name = f"commons__{got + existing:02d}.jpg"
                target = scene_dir / name
                if str(target) in seen:
                    continue
                try:
                    request = urllib.request.Request(
                        thumb, headers={"User-Agent": USER_AGENT}
                    )
                    with urllib.request.urlopen(request, timeout=30) as response:
                        target.write_bytes(response.read())
                except OSError:
                    continue
                attribution.append({
                    "file": f"{scene.id}/{name}",
                    "scene": scene.id,
                    "title": page.get("title", ""),
                    "page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page.get('title', ''))}",
                    "license": license_name,
                    "artist": extmeta.get("Artist", {}).get("value", ""),
                    "query": query,
                })
                seen.add(str(target))
                got += 1
                # Atrybucja zapisywana przyrostowo — restart nie gubi metadanych.
                attribution_path.write_text(
                    json.dumps(attribution, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                time.sleep(0.8)
        print(f"{scene.id}: {got} nowych (razem {existing + got})", flush=True)

    attribution_path.write_text(
        json.dumps(attribution, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Atrybucja zapisana: {attribution_path} ({len(attribution)} plików)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pobiera zdjęcia z Wikimedia Commons")
    parser.add_argument("--photos", default="photos")
    parser.add_argument("--per-scene", type=int, default=4)
    parser.add_argument("--only", default=None, help="filtr po id sceny (substring), np. opakowania/")
    args = parser.parse_args()
    fetch(args.per_scene, Path(args.photos), only=args.only)


if __name__ == "__main__":
    main()
