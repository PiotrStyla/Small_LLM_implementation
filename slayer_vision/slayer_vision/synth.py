"""Syntetyczne obrazy dokumentów i paneli Z PEŁNYM ground truth.

Renderujemy proste, ale realistycznie ułożone sceny tekstowe: rachunek,
etykieta z datą ważności, blister leków, panel AGD z przyciskami, pilot,
list/pismo, metka z ceną. Każde zdanie jest generowane z tych samych slotów,
które rysujemy — GT jest dokładne (ten sam patent co w syntezie PolOCRBench).

Ważne (wniosek z ewaluacji run-800): wartości slotów MUSZĄ wędrować po
obrazie — stałe pozycje prowadzą do „mode collapse" (model zwraca najczęstszą
kwotę zamiast czytać). Każdy renderer ma dżitter pozycji/rozmiarów fontu
i szerokie pule slotów.

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

# Szerokie pule slotów — im więcej wariantów, tym lepiej model czyta,
# zamiast zgadywać najczęstszą wartość.
SHOPS = (
    "Biedronka", "Żabka", "Lidl", "Auchan", "Stokrotka", "Carrefour",
    "Dino", "Kaufland", "Netto", "Lewiatan", "Polo Market", "Intermarche",
)
EXPIRY_PRODUCTS = (
    ("MLEKO", "mleka"), ("JOGURT", "jogurtu"), ("SER", "sera"),
    ("LEK", "leku"), ("SOK", "soku"), ("WEDLINY", "wędliny"),
    ("MASLO", "masła"), ("KECZUP", "keczupu"), ("MIOD", "miodu"),
    ("SEREK", "serka"), ("MAKARON", "makaronu"), ("PRZYPRAWA", "przyprawy"),
    ("DZEM", "dżemu"), ("SMIETANA", "śmietany"),
)
PRICE_PRODUCTS = (
    "mleko", "chleb", "masło", "ser", "kawa", "herbata", "jogurt", "makaron",
    "ryż", "sok", "czekolada", "dżem",
)
PILL_NAMES = (
    "Apap", "Ibuprom", "Rutinoscorbin", "Polopiryna", "Magnez", "Calcium",
    "Witamina C", "Acard", "Euthyrox", "Metformax",
)
LETTER_SENDERS = ("banku", "ZUS", "urzędu miasta", "przychodni", "operatora prądu")
LETTER_TYPES = ("list", "wezwanie do zapłaty", "decyzja", "zawiadomienie")


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
    return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2025, 2030)}"


def _jitter(rng: random.Random, x: int, y: int, spread: int = 14) -> tuple[int, int]:
    """Dżitter pozycji — wartości slotów nie mogą „siedzieć" w stałym miejscu."""
    return x + rng.randint(-spread, spread), y + rng.randint(-spread // 2, spread // 2)


def _size(rng: random.Random, base: int, spread: int = 5) -> int:
    return max(11, base + rng.randint(-spread, spread))


# --- renderery: (rng) -> (Image, zdanie PL) ---------------------------------

def receipt(rng: random.Random) -> tuple[Image.Image, str]:
    sklep = rng.choice(SHOPS)
    kwota = _money(rng)
    items = rng.randint(3, 7)
    img, draw = _paper()
    draw.rectangle([24, 16, 360, 368], outline="#444", width=2)
    draw.text(_jitter(rng, 48, 28), sklep.upper(), font=_font(_size(rng, 30), bold=True), fill="black")
    draw.text((48, 80), "PARAGON FISKALNY", font=_font(_size(rng, 18)), fill="black")
    for i in range(items):
        draw.text((48, 118 + i * 24), f"ARTYKUL {i + 1}", font=_font(15), fill="#333")
        draw.text((276, 118 + i * 24), f"{rng.randint(1, 40)},{rng.randint(0, 99):02d}", font=_font(15), fill="#333")
    draw.line([48, 118 + items * 24 + 6, 336, 118 + items * 24 + 6], fill="black", width=2)
    draw.text(_jitter(rng, 48, 292), "DO ZAPLATY", font=_font(_size(rng, 24), bold=True), fill="black")
    draw.text(_jitter(rng, 232, 292), f"{kwota} zl", font=_font(_size(rng, 24), bold=True), fill="black")
    return img, f"To paragon ze sklepu {sklep}. Do zapłaty {kwota} zł."


def expiry_label(rng: random.Random) -> tuple[Image.Image, str]:
    label, genitive = rng.choice(EXPIRY_PRODUCTS)
    data = _date(rng)
    img, draw = _paper()
    draw.rectangle([28, 100, 356, 284], outline="#222", width=3)
    draw.text(_jitter(rng, 48, 122), label, font=_font(_size(rng, 26), bold=True), fill="black")
    draw.text(_jitter(rng, 48, 174), "NAJLEPIEJ SPOZYC", font=_font(_size(rng, 18)), fill="#333")
    draw.text(_jitter(rng, 48, 212, spread=20), data, font=_font(_size(rng, 34, 7), bold=True), fill="black")
    return img, f"Data ważności na opakowaniu {genitive} to {data}."


def pill_blister(rng: random.Random) -> tuple[Image.Image, str]:
    nazwa = rng.choice(PILL_NAMES)
    rows, cols = rng.choice([(2, 4), (2, 5), (3, 4), (2, 3)])
    total = rows * cols
    left = rng.randint(1, total - 1)
    img, draw = _paper()
    draw.rectangle([16, 40, 368, 344], fill="#dfe6ee", outline="#8899aa", width=2)
    draw.text(_jitter(rng, 28, 10), f"{nazwa} — tabletki", font=_font(_size(rng, 20), bold=True), fill="black")
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
    draw.text(_jitter(rng, 28, 20), f"Panel {device}", font=_font(_size(rng, 22), bold=True), fill="black")
    w = 330 // len(buttons)
    for i, label in enumerate(buttons):
        x = 24 + i * w
        draw.rounded_rectangle([x, 130, x + w - 16, 250], radius=12, fill="#3a3f44")
        draw.text((x + 12, 170), label, font=_font(_size(rng, 18), bold=True), fill="white")
    draw.text((28, 280), "wybierz program", font=_font(16), fill="#333")
    place = ["pierwszy od lewej", "środkowy", "ostatni od lewej"][hit]
    return img, f"To panel {device}. Przycisk „{buttons[hit]}” jest {place}."


def remote(rng: random.Random) -> tuple[Image.Image, str]:
    brand = rng.choice(["Samsung", "LG", "Sony", "Philips", "Panasonic", "Toshiba"])
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "#e8e4dc")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([124, 12, 260, 372], radius=28, fill="#23262b")
    draw.text(_jitter(rng, 150, 24, spread=8), brand, font=_font(_size(rng, 18), bold=True), fill="#ddd")
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
    nadawca = rng.choice(LETTER_SENDERS)
    typ = rng.choice(LETTER_TYPES)
    kwota = _money(rng) if "zapłat" in typ else None
    img, draw = _paper()
    draw.rectangle([20, 20, 364, 364], outline="#777", width=1)
    draw.text(_jitter(rng, 44, 42), f"{typ.upper()}", font=_font(_size(rng, 24), bold=True), fill="black")
    for i in range(6):
        draw.line([44, 108 + i * 34, rng.randint(220, 340), 108 + i * 34], fill="#999", width=3)
    if kwota:
        draw.text(_jitter(rng, 44, 296, spread=20), f"Do zapłaty: {kwota} zł", font=_font(_size(rng, 24), bold=True), fill="black")
        sentence = f"To wezwanie do zapłaty z {nadawca}. Do zapłaty {kwota} zł."
    else:
        sentence = f"To {typ} z {nadawca}."
    return img, sentence


def price_tag(rng: random.Random) -> tuple[Image.Image, str]:
    produkt = rng.choice(PRICE_PRODUCTS)
    cena = _money(rng)
    img, draw = _paper()
    draw.rectangle([40, 80, 344, 304], fill="white", outline="#222", width=3)
    draw.text(_jitter(rng, 64, 108), produkt.upper(), font=_font(_size(rng, 28), bold=True), fill="black")
    draw.text(_jitter(rng, 64, 170, spread=20), f"{cena} zl / szt", font=_font(_size(rng, 38, 7), bold=True), fill="#8a1f1f")
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
