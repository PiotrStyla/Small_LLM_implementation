"""Build human-annotated object-caption pairs from COCO 2017 bounding boxes.

Images are separate Flickr works. Keep only images whose COCO per-image
license explicitly permits adaptation (CC BY or CC BY-SA); preserve source,
license and box in attribution.jsonl. Never redistribute the source photos as
part of the model or claim that the labels cover an entire camera scene.

Example:
    python -m slayer_vision.build_coco_objects \
        --annotations-zip out/annotations_trainval2017.zip \
        --out out/coco-objects --train-per-class 16 --eval-per-class 4 \
        --reviewed-val coco_val_reviewed.txt
"""

import argparse
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import ijson
import requests
from PIL import Image, UnidentifiedImageError

CAPTIONS = {
    "bottle": "To butelka.",
    "cup": "To kubek.",
    "chair": "To krzesło.",
    "couch": "To kanapa.",
    "bed": "To łóżko.",
    "tv": "To telewizor.",
    "mouse": "To mysz komputerowa.",
    "remote": "To pilot.",
    "keyboard": "To klawiatura.",
    "clock": "To zegar.",
    "book": "To książka.",
    "cell phone": "To telefon.",
    "vase": "To wazon.",
}


def _items(archive: zipfile.ZipFile, name: str, prefix: str):
    with archive.open(f"annotations/instances_{name}2017.json") as source:
        yield from ijson.items(source, prefix)


def _licensed(url: str) -> bool:
    return "creativecommons.org/licenses/by/" in url or "creativecommons.org/licenses/by-sa/" in url


def _candidates(archive: zipfile.ZipFile, split: str):
    licenses = {item["id"]: item for item in _items(archive, split, "licenses.item")}
    categories = {
        item["id"]: item["name"] for item in _items(archive, split, "categories.item")
    }
    images = {
        item["id"]: item
        for item in _items(archive, split, "images.item")
        if _licensed(licenses[item["license"]]["url"])
    }
    selected = defaultdict(list)
    for annotation in _items(archive, split, "annotations.item"):
        image = images.get(annotation["image_id"])
        name = categories[annotation["category_id"]]
        if image is None or name not in CAPTIONS or annotation.get("iscrowd"):
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        if (
            width < 80 or height < 80
            or width * height < 0.08 * image["width"] * image["height"]
        ):
            continue
        selected[name].append((image, [x, y, width, height], licenses[image["license"]]))
    return selected


def _download_crop(session: requests.Session, image: dict, box: list, split: str):
    # COCO's S3 bucket offers a TLS-valid path-style URL. Do not disable TLS.
    url = f"https://s3.amazonaws.com/images.cocodataset.org/{split}2017/{image['file_name']}"
    response = session.get(url, timeout=40)
    response.raise_for_status()
    with Image.open(io.BytesIO(response.content)) as original:
        rgb = original.convert("RGB")
        x, y, width, height = box
        margin = 0.18 * max(width, height)
        bounds = (
            max(0, int(x - margin)), max(0, int(y - margin)),
            min(rgb.width, int(x + width + margin)),
            min(rgb.height, int(y + height + margin)),
        )
        return rgb.crop(bounds)


def build(annotations_zip: Path, out: Path, train_per_class: int, eval_per_class: int,
          seed: int, reviewed_val: Path | None = None):
    (out / "images").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    attribution = []
    rng = random.Random(seed)
    with zipfile.ZipFile(annotations_zip) as archive:
        for split, amount in (("train", train_per_class), ("val", eval_per_class)):
            candidates = _candidates(archive, split)
            records = []
            used_images = set()
            for name, sentence in CAPTIONS.items():
                choices = candidates[name]
                rng.shuffle(choices)
                count = 0
                for image, box, license_info in choices:
                    if count >= amount:
                        break
                    if image["id"] in used_images:
                        continue
                    try:
                        crop = _download_crop(session, image, box, split)
                    except (requests.RequestException, OSError, UnidentifiedImageError) as error:
                        print(f"Pominięto {image['id']}: {error}", flush=True)
                        continue
                    filename = f"{split}_{image['id']}_{name.replace(' ', '_')}.jpg"
                    crop.save(out / "images" / filename, quality=90)
                    records.append({"image": filename, "text": sentence, "scene": name})
                    attribution.append({
                        "file": filename, "image_id": image["id"], "scene": name,
                        "flickr_url": image["flickr_url"], "coco_url": image["coco_url"],
                        "license": license_info["name"], "license_url": license_info["url"],
                        "bbox": box, "split": split,
                    })
                    used_images.add(image["id"])
                    count += 1
                print(f"{split} {name}: {count}/{amount}", flush=True)
                if (split == "train" and count != amount) or (split == "val" and count < 2):
                    raise RuntimeError(f"Za mało obrazów dla {split}/{name}: {count}/{amount}")
            target = "train.jsonl" if split == "train" else "eval.jsonl"
            with (out / target).open("w", encoding="utf-8") as destination:
                for row in records:
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "attribution.jsonl").open("w", encoding="utf-8") as destination:
        for row in attribution:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    if reviewed_val is not None:
        approved = {
            int(line.strip()) for line in reviewed_val.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        eval_rows = [
            json.loads(line) for line in (out / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        reviewed = [
            row for row in eval_rows if int(row["image"].split("_")[1]) in approved
        ]
        if {int(row["image"].split("_")[1]) for row in reviewed} != approved:
            raise ValueError("Lista ręcznie sprawdzonych zdjęć nie pasuje do podziału val")
        scenes = {row["scene"] for row in reviewed}
        train_rows = [
            json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for filename, records in (
            ("eval-reviewed.jsonl", reviewed),
            ("train-target-classes.jsonl", [row for row in train_rows if row["scene"] in scenes]),
        ):
            with (out / filename).open("w", encoding="utf-8") as destination:
                for row in records:
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Ręcznie sprawdzone: {len(reviewed)} val, {len(scenes)} klas", flush=True)
    print(f"Zapisano {len(attribution)} licencjonowanych kadrów: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=16)
    parser.add_argument("--eval-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reviewed-val", type=Path)
    args = parser.parse_args()
    build(
        args.annotations_zip, args.out, args.train_per_class, args.eval_per_class,
        args.seed, reviewed_val=args.reviewed_val,
    )


if __name__ == "__main__":
    main()
