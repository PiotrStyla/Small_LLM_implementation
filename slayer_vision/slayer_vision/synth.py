"""Syntetyczne obrazy dokumentów i paneli Z PEŁNYM ground truth.

Renderujemy proste, ale realistycznie ułożone sceny tekstowe: rachunek,
etykieta z datą ważności, blister leków, panel AGD, pilot, list, metka z ceną.
Każdy renderer zwraca (obraz PIL, zdanie PL ze slotami) — zdanie jest
generowane z tych samych wartości, które rysujemy, więc GT jest dokładne
(ten sam patent co w syntezie PolOCRBench/OCR_engine).

Fonty: TrueType jest wymagany, bo polskie znaki nie przejdą przez domyślny
font bitmapowy PIL-a.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = 384  # CaptionDataset i tak skaluje do 224 — renderujemy z zapasem.

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["C:/Windows/Fonts/arialbd.ttf", *_FONT_CANDIDATES] if bold else _FONT_CANDIDATES
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError(
        "Brak fontu TrueType (Arial/DejaVu) — polskie znaki wymagają TTF."
    )


def _paper() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "#f4f1ea")
    return img, ImageDraw.Draw(img)


def _money(rng: random.Random) -> str:
    return f"{rng.randint(3, 249)},{rng.randint(0, 99):02d}"


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2026, 2028)}"


# --- renderery: (rng) -> (Image, zdanie PL) ---------------------------------

def receipt(rng: random.Random) -> tuple[Image.Image, str]:
    sklep = rng.choice(["Biedronka", "Żabka", "Lidl", "Auchan", "Stokrotka"])
    kwota = _money(rng)
    img, draw = _paper()
    draw.rectangle([24, 16, 360, 368], outline="#444", width=2)
    draw.text((48, 32), sklep.upper(), font=_font(30, bold=True), fill="black")
    draw.text((48, 80), "PARAGON FISKALNY", font=_font(18), fill="black")
    for i in range(5):
        draw.text((48, 122 + i * 28), f"ARTYKUL {i + 1}", font=_font(16), fill="#333")
        draw.text((280, 122 + i * 28), f"{rng.randint(1, 40)},{rng.randint(0, 99):02d}", font=_font(16), fill="#333")
    draw.line([48, 272, 336, 272], fill="black", width=2)
    draw.text((48, 290), "DO ZAPLATY", font=_font(24, bold=True), fill="black")
    draw.text((240, 290), f"{kwota} zl", font=_font(24, bold=True), fill="black")
    return img, f"To paragon ze sklepu {sklep}. Do zapłaty {kwota} zł."


def expiry_label(rng: random.Random) -> tuple[Image.Image, str]:
    produkt = rng.choice(["mleko", "jogurt", "ser", "lek", "sok", "wędliny"])
    data = _date(rng)
    img, draw = _paper()
    draw.rectangle([28, 100, 356, 284], outline="#222", width=3)
    draw.text((48, 122), produkt.upper(), font=_font(26, bold=True), fill="black")
    draw.text((48, 176), "NAJLEPIEJ SPOZYC", font=_font(18), fill="#333")
    draw.text((48, 214), data, font=_font(34, bold=True), fill="black")
    return img, f"Data ważności na opakowaniu {produkt} to {data}."


def pill_blister(rng: random.Random) -> tuple[Image.Image, str]:
    nazwa = rng.choice(["Apap", "Ibuprom", "Rutinoscorbin", "Polopiryna", "Magnez"])
    rows, cols = rng.choice([(2, 4), (2, 5), (3, 4)])
    total = rows * cols
    left = rng.randint(1, total - 1)
    img, draw = _paper()
    draw.rectangle([16, 40, 368, 344], fill="#dfe6ee", outline="#8899aa", width=2)
    draw.text((28, 12), f"{nazwa} — tabletki", font=_font(20, bold=True), fill="black")
    w, h = 320 // cols, 250 // rows
    for r in range(rows):
        for c in range(cols):
            x, y = 32 + c * w, 60 + r * h
            if r * cols + c < left:
                draw.ellipse([x + 4, y + 4, x + w - 8, y + h - 8], fill="#f7f7f2", outline="#99a")
    return img, f"To blister leku {nazwa}. Zostało {left} tabletek."


def device_panel(rng: random.Random) -> tuple[Image.Image, str]:
    device = rng.choice(["pralki", "zmywarki", "piekarnika"])
    buttons = rng.choice([
        ["WŁĄCZ", "START", "PROGRAM"],
        ["START", "STOP", "OPÓŹNIENIE"],
        ["PROGRAM", "START", "TEMPERATURA"],
    ])
    hit = rng.randrange(len(buttons))
    img, draw = _paper()
    draw.rectangle([12, 60, 372, 324], fill="#cfd4d8", outline="#555", width=3)
    draw.text((28, 22), f"Panel {device}", font=_font(22, bold=True), fill="black")
    w = 330 // len(buttons)
    for i, label in enumerate(buttons):
        x = 24 + i * w
        draw.rounded_rectangle([x, 130, x + w - 16, 250], radius=12, fill="#3a3f44")
        draw.text((x + 12, 170), label, font=_font(18, bold=True), fill="white")
    draw.text((28, 280), "wybierz program", font=_font(16), fill="#333")
    place = ["pierwszy od lewej", "środkowy", "ostatni od lewej"][hit]
    return img, f"To panel {device}. Przycisk „{buttons[hit]}” jest {place}."


def remote(rng: random.Random) -> tuple[Image.Image, str]:
    brand = rng.choice(["Samsung", "LG", "Sony", "Philips"])
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "#e8e4dc")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([124, 12, 260, 372], radius=28, fill="#23262b")
    draw.text((150, 26), brand, font=_font(18, bold=True), fill="#ddd")
    draw.ellipse([150, 60, 234, 122], fill="#8a1f1f")  # power
    draw.text((158, 78), "WŁ", font=_font(20, bold=True), fill="white")
    labels = ["VOL+", "VOL-", "CH+", "CH-"]
    for i, label in enumerate(labels):
        y = 150 + (i // 2) * 66
        x = 140 + (i % 2) * 68
        draw.rounded_rectangle([x, y, x + 58, y + 50], radius=8, fill="#4a4f56")
        draw.text((x + 8, y + 14), label, font=_font(15, bold=True), fill="white")
    draw.ellipse([162, 300, 222, 350], fill="#4a4f56")
    return img, f"To pilota marki {brand}. Przycisk zasilania jest na górze po lewej stronie."


def letter(rng: random.Random) -> tuple[Image.Image, str]:
    nadawca = rng.choice(["banku", "ZUS", "urzędu miasta", "przychodni", "operatora prądu"])
    typ = rng.choice(["list", "wezwanie do zapłaty", "decyzja", "zawiadomienie"])
    kwota = _money(rng) if "zapłat" in typ else None
    img, draw = _paper()
    draw.rectangle([20, 20, 364, 364], outline="#777", width=1)
    draw.text((44, 44), f"{typ.upper()}", font=_font(24, bold=True), fill="black")
    for i in range(6):
        draw.line([44, 108 + i * 34, rng.randint(220, 340), 108 + i * 34], fill="#999", width=3)
    if kwota:
        draw.text((44, 300), f"Do zapłaty: {kwota} zł", font=_font(24, bold=True), fill="black")
        sentence = f"To wezwanie do zapłaty z {nadawca}. Do zapłaty {kwota} zł."
    else:
        sentence = f"To {typ} z {nadawca}."
    return img, sentence


def price_tag(rng: random.Random) -> tuple[Image.Image, str]:
    produkt = rng.choice(["mleko", "chleb", "masło", "ser", "kawa", "herbata"])
    cena = _money(rng)
    img, draw = _paper()
    draw.rectangle([40, 80, 344, 304], fill="white", outline="#222", width=3)
    draw.text((64, 110), produkt.upper(), font=_font(28, bold=True), fill="black")
    draw.text((64, 172), f"{cena} zl / szt", font=_font(38, bold=True), fill="#8a1f1f")
    draw.text((64, 248), "cena detaliczna", font=_font(16), fill="#555")
    return img, f"Cena produktu {produkt} wynosi {cena} zł."


RENDERERS = {
    "rachunek": receipt,
    "data_waznosci": expiry_label,
    "blister": pill_blister,
    "panel": device_panel,
    "pilot": remote,
    "list": letter,
    "cena": price_tag,
}


def render(kind: str, seed: int) -> tuple[Image.Image, str]:
    """Renderuje scenę [kind] deterministycznie dla [seed] → (obraz, zdanie)."""
    if kind not in RENDERERS:
        raise KeyError(f"Nieznany typ sceny syntetycznej: {kind}")
    return RENDERERS[kind](random.Random(seed))
