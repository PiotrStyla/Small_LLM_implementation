# Small LLM implementation — Asystent wzrokowy (on-device)

Eksperymentalny asystent wzrokowy dla osób niewidomych i seniorów. Aparat
rejestruje zdjęcie, a telefon pokazuje i wypowiada po polsku opis lub odczytany
tekst. **Nie jest to bezpieczny zamiennik samodzielnej oceny otoczenia**:
model myli przedmioty, OCR myli cyfry, a aplikacja bywa zamykana przy braku
pamięci. Nie używać jej jako jedynego źródła decyzji o ruchu, lekach,
terminach ważności ani płatnościach.

```
opis: zdjęcie → SigLIP → projector → mały polski LLM goLLeM → podpis → TTS
tekst/data/kwota: zdjęcie → lokalny OCR ML Kit → ostrożny odczyt → TTS
```

## Możliwości (MVP)

- 📸 **Jeden duży przycisk „Pokaż i zapytaj"** — zdjęcie + domyślne pytanie
  „Co jest przede mną?".
- **„Co to jest?"** — podpis obrazu SLAYER lub pytanie do Gemma 3n.
- **Polecenia tekstowe**: „Przeczytaj tekst" (cały odczyt), „Data ważności"
  (tylko data obok oznaczenia terminu), „Ile do zapłaty?" (kwota oznaczona na
  paragonie) — lokalny OCR ML Kit; bez tekstu nie zgaduje cyfr.
- **Który przycisk?**: szuka czytelnego napisu zasilania (np. POWER/ON-OFF);
  nie rozpoznaje samej ikony ani nie wskazuje położenia przycisku.
- **Pytania głosem** (systemowy ASR, polski); w trybie SLAYER obsługiwane są
  wyłącznie rozpoznane polecenia z powyższego katalogu.
- **Odpowiedź głosowa** przez systemowy TTS + duży tekst (odsłuch na telefonie
  niezweryfikowany); rozpoznawanie pytań głosowych korzysta z systemowego ASR.
- **Offline dla obrazu, OCR i generacji SLAYER** po instalacji plików modelu;
  zdjęcia nie są wysyłane przez aplikację. Systemowe ASR może wymagać sieci
  lub przetwarzać głos przez usługę systemową — nie gwarantujemy offline ASR.
- Etykiety dostępności Semantics dla TalkBack/VoiceOver i duże przyciski;
  użyteczność z użytkownikami niewidomymi nie została przetestowana.

## Stack

| Warstwa | Wybór | Dlaczego |
|---|---|---|
| App | Flutter (Android + iOS, jeden kod) | flutter_gemma jest natywnie multiplatformowy |
| Model | **SLAYER-Vision-PL** (ONNX) lub Gemma 3n E2B (`.litertlm`) | własny mały model opisujący zdjęcia albo model ogólny odpowiadający na pytania |
| Inference | `onnxruntime` lub `flutter_gemma` + `flutter_gemma_litertlm` | lokalnie na Androidzie i iOS |
| Tekst na zdjęciu | ML Kit Latin (model dołączony do aplikacji) | OCR na pełnym zdjęciu, lokalnie |
| TTS / ASR | `flutter_tts` / `speech_to_text` | systemowe — zero dodatkowych modeli, znajomy głos |
| Kamera | `camera` | standard Fluttera |

SLAYER-Vision-PL (SigLIP-base + projector + goLLeM-110M-PL-SFT) opisuje zdjęcie
jednym zdaniem i **nie rozumie pytania**. Opis może mylić obiekty, zwłaszcza
na rozmazanych zdjęciach. Zadania tekstowe aplikacja kieruje do OCR zamiast
udawać, że model przeczytał datę lub kwotę. Dowolne pytania o zdjęcie
wymagają Gemma 3n.

## SLAYER-Vision-PL

**Mały LLM ma konkretną rolę:** `SlayerLab/goLLeM-110M-PL-SFT-merged` to
polski decoder-only GPT-2 (~110 mln parametrów, 12 warstw, kontekst 512).
Nie ogląda pikseli bezpośrednio. Zamrożony SigLIP-base (~86 mln parametrów)
przekształca zdjęcie 224×224 w 196 wektorów; trenowany projector (~3,15 mln)
rzutuje je do przestrzeni goLLeM. Dostrojenie LoRA (~0,59 mln) scalono
z modelem językowym przy eksporcie; łącznie ~206,6 mln parametrów.

```
obraz 224×224 → SigLIP-base → 196 wektorów → projector
→ wejście goLLeM-110M-PL-SFT (+ scalona LoRA) → podpis obrazu po polsku
```

Model trenowano do **podpisywania zdjęć**, bez promptu tekstowego i BOS.
Generacja zaczyna się od wektorów obrazu. Naciśnięcie „Co jest przede mną?”
lub „Co to jest?” wywołuje ten sam rodzaj podpisu: LLM **nie otrzymuje treści
pytania**. Pozostałe przyciski przełączają na OCR, a nie na rozumienie poleceń
przez LLM. To nie jest VQA ani ogólna rozmowa o obrazie.

| Metryka | Wartość |
|---|---|
| Parametry łącznie | **206,6 M** (zmierzone w `smoke_test.py`) |
| Trenowalne (projector + LoRA r=16) | **3,74 M** |
| Budżet kontekstu | 196 (obraz) + tekst + odpowiedź ≤ 512 (ctx GoLLeM) |
| Format danych | JSONL: `{"image": "plik.jpg", "text": "Krótkie zdanie PL."}` |

Kod: `slayer_vision/` — `model.py` (okablowanie), `data.py` (dataset),
`train.py` (pętla treningowa), `smoke_test.py` (dowód na CPU).

```bash
cd slayer_vision
python -m pip install -r requirements.txt
python smoke_test.py
python -m slayer_vision.build_dataset --out out/dataset-safe --per-kind 40 --photos photos
```

`build_dataset` domyślnie **pomija nieprzejrzane zdjęcia Commons**. Ich
automatyczne przypisanie na podstawie wyszukiwania okazało się błędne; dopiero
lista zatwierdzonych plików `--reviewed-photos photos/approved.txt` dopuszcza
je do treningu. Każdy wiersz listy ma postać `kategoria/scena/commons__NN.jpg`;
zatwierdzaj tylko zdjęcie zgodne z opisem, nie na podstawie nazwy katalogu.

### Eksport on-device (ONNX)

flutter_gemma nie obsługuje własnej architektury SigLIP + projector + goLLeM;
aplikacja składa ją z trzech grafów ONNX:

```bash
python -m slayer_vision.export_onnx --checkpoint out/run-mixed-long --out out/export-slayer-vision-v2
```

| Plik | Rola |
|---|---|
| `vision_projector.onnx` (+`.data`) | piksele → 196 tokenów obrazu (batch 1) |
| `lm_embeds.onnx` (+`.data`) | embeddy + maska → logits; **sekwencja dynamiczna** |
| `embed_tokens.onnx` (+`.data`) | id wygenerowanych tokenów → embeddy kolejnego kroku |
| `tokens_decoded.json` | mapa id → tekst (aplikacja tylko dekoduje — format treningowy nie ma promptu/BOS) |
| `manifest.json` | wymiary, pliki, wynik testu parzystości |

Weryfikacja przy eksporcie: **test dekodowania greedy PyTorch ↔ ONNX** — musi
wyjść identyczne zdanie (akceptacja produktowa; same różnice liczbowe fp32
rzędu 1e-3 to szum przekształceń grafu). Eksporter używa opsetu 18 (starszy
psuje `Split`) i zapisuje IR 9 zgodny z mobilnym ONNX Runtime; IR 10 z domyślnego
eksportu nie ładuje się na urządzeniu. Wszystkie pliki `*.onnx.data` muszą leżeć
obok grafów.

Grafy i wagi zajmują ~897 MB fp32. Kwantyzacja int8 nie jest jeszcze wdrożona.

**Ewaluacja i odporność:** `evaluate.py` generuje zdanie do każdego obrazu
z `eval.jsonl` (greedy, start z samych tokenów obrazu — format treningowy nie
ma BOS) i liczy trafienie dokładne po normalizacji + F1 tokenów. Flaga
`--degrade` powtarza pomiar na zdegradowanych obrazach (deterministycznie per
próbka) — różnica to miara odporności na warunki zdjęcia. W treningu `--augment`
dokłada losowe degradacje (`augment.py`: jasność, kontrast, rozmycie, szum,
obrót ±6°, kwantyzacja zamiennikiem JPEG).

Zapis checkpointu: `projector.pt` + adaptery LoRA (`lora/`) + manifest
`slayer_vision.json` — do dołożenia w aplikacji zamiast Gemma 3n, gdy tor
treningowy domknie jakość.

### Zbiór danych (katalog 200 scen)

Zamknięty katalog w `slayer_vision/scenes.py` — **200 scen** w 10 kategoriach
(opakowania, leki, AGD/panele, piloty, dokumenty, zakupy, ulica, dom,
przesyłki, przedmioty codzienne). Zamknięcie katalogu jest świadomym
kompromisem: 110M model halucynuje w otwartym „opisz świat", a użytkownik
potrzebuje przewidywalnych, jednozdaniowych odpowiedzi.

Dwa źródła próbek:

1. **Syntetyka z pełnym GT** (`synth.py`) — 7 rendererów PIL: rachunek/paragon,
   etykieta z datą ważności, blister leków, panel AGD z przyciskami, pilot,
   list/pismo, metka z ceną. Zdanie generowane z tych samych slotów, które
   są rysowane (kwota, data, nazwa leku, położenie przycisku) — GT dokładne
   co do grosza, jak w syntezie PolOCRBench.
2. **Zdjęcia:** w `photos/` znajdują się pobrane miniatury Commons z
   `photos/ATTRIBUTION.json`, ale **nie są automatycznie czystym zbiorem**.
   Kontrola wizualna ujawniła w klasie „karton soku” drogę, żabę i owady,
   a w „karton mleka” m.in. szklankę mleka. Próbny przebieg buildera pominął
   **785/785** takich niezweryfikowanych zdjęć. Własne poprawnie opisane kadry
   można dodawać do `photos/<scene_id>/*.jpg`; obce zdjęcia `commons__*`
   wymagają ręcznego zatwierdzenia w `--reviewed-photos`.

Stary checkpoint `run-final` wytrenowano na danych sprzed tej kontroli.
Wynik **114/319 exact** na dawnym `eval.jsonl` jest wynikiem dla
**zanieczyszczonego, częściowo źle oznaczonego zbioru**, a nie miarą
rozpoznawania rzeczywistych przedmiotów.

### Niezależny zbiór obiektów COCO 2017

`build_coco_objects.py` używa anotacji prostokątów obiektów z
[COCO 2017](https://cocodataset.org/dataset/detection-2017.htm):
przycina obiekt, przypisuje krótkie polskie zdanie, rozdziela obrazy
`train2017` i `val2017` i zachowuje URL źródłowy, prostokąt oraz
**indywidualną licencję zdjęcia** w `attribution.jsonl`. Wybiera tylko
zdjęcia oznaczone CC BY lub CC BY-SA; adnotacje COCO to
[CC BY 4.0](https://cocodataset.org/#termsofuse). Źródłowe zdjęcia nie
są częścią repozytorium. Lista `coco_val_reviewed.txt` zawiera identyfikatory
38 ręcznie przejrzanych zdjęć 12 klas; **treningowe kadry opierają się na
anotacjach COCO, nie przeszły indywidualnej kontroli wizualnej**.
To test wybranych obiektów na przyciętych fotografiach, nie test telefonu
na ulicy ani poprawności kartonów soku lub soli (COCO nie obejmuje ich tu).

```bash
cd slayer_vision
curl -fL https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip \
    -o out/annotations_trainval2017.zip
python -m slayer_vision.build_coco_objects \
    --annotations-zip out/annotations_trainval2017.zip --out out/coco-objects \
    --train-per-class 16 --eval-per-class 4 --reviewed-val coco_val_reviewed.txt
python -m slayer_vision.evaluate --checkpoint out/run-final \
    --data-jsonl out/coco-objects/eval-reviewed.jsonl \
    --image-root out/coco-objects/images
```

`out/coco-objects/train-target-classes.jsonl` zawiera 192 kadry, a
`eval-reviewed.jsonl` 38 niezależnych zdjęć. Przed dalszym treningiem
stary model uzyskał na nich **9/38 exact (23,7%)**, F1 tokenów **0,613**.

### Dalszy trening — rozpoznawanie obiektów (2026-09-25)

Trzy przebiegi na CPU (`torch 2.13`, batch 1, wznawiane z `run-final`),
każdy oceniany na tych samych 38 zdjęciach:

| Checkpoint | Kroki | Dane | Obiekty exact | Obiekty F1 |
|---|---:|---|---:|---:|
| `run-final` | 3600 | stary zbiór (zanieczyszczony) | 9/38 (23,7%) | 0,613 |
| `run-coco-objects` | 60 | 192 kadry COCO | 24/38 (63,2%) | 0,808 |
| `run-mixed` | +120 | COCO + 488 zdań syntetycznych | 29/38 (76,3%) | 0,876 |
| `run-mixed-long` | +300 | jw., LR 8e-5/1,5e-5 | **31/38 (81,6%)** | **0,913** |

Wzrost jest nierówny per klasa (`run-mixed-long` vs baseline): zegar 0/4→4/4,
telewizor 0/4→4/4, butelka 0/3→3/3, kubek 1/3→3/3, telefon 2/4→4/4,
książka 3/4→4/4, klawiatura 0/4→3/4, łóżko 1/2→2/2, krzesło 0/2→1/2;
słabe punkty: pilot 0/2, kanapa 1/3, mysz 2/3. Pozostałe pomyłki to sąsiednie
klasy (kanapa↔łóżko↔krzesło, pilot→telewizor/telefon, mysz→klawiatura).
**n=38 — przedziały niepewności są szerokie; to nie jest skuteczność uliczna.**

Dwa skutki uboczne, zmierzone a nie założone:

1. **Krótkie podpisy COCO ucinają zdania na zdjęciach realnych.** Po
   samym treningu COCO model zaczął kończyć po dwóch słowach
   („To kostka masła.” → „To kostka.”; 34/319 ucięć vs 2/319 w baseline).
   Domykanie długimi zdaniami syntetycznymi naprawiło zdania dokumentów
   (probe 20/20 exact, pełne zdania) i obiekty, ale na zdjęciach realnych
   kwalifikatory nadal bywają ucinane („To butelka wody mineralnej.” →
   „To butelka.”). Nazwa obiektu pozostaje poprawna; model mówi krócej.
2. **Spadek exact na starym `eval.jsonl` jest częściowo pozorny.** Rozbicie
   `run-final` → `run-mixed-long` na tym samym zbiorze:
   - syntetyka (pewne GT): **31/122 → 34/122 exact, F1 0,794 → 0,829** — lepiej;
   - zdjęcia (etykiety błędne): 83/197 → 60/197 exact, F1 0,642 → 0,643 —
     exact spada głównie przez ucinanie kwalifikatorów i odejście od złych
     etykiet; to nie jest rzetelny pomiar jakości.

Tekst w aplikacji czyta **osobny OCR ML Kit** — żaden z tych przebiegów nie
zmienia ścieżki OCR ani reguł dat/kwot.

### Druga iteracja — v0.3 (2026-09-25)

Dane rozbudowane: klas obiektów 13 → **40** (jedzenie, kuchnia, dom, ulica),
kadrów treningowych 192 → **1271** (32/klasę), syntetyka 488 → **660** (80/typ,
seed 43). Nowy benchmark **82 przejrzanych zdjęć 27 nowych klas**
(`coco_val_reviewed2.txt`; 22/104 odrzucone przy recenzji, m.in. hot-dog jako
„kanapka”, szczypce jako „nożyczki”). Trening: 500 kroków (LR 8e-5/1,5e-5)
+ 1000 kroków (LR 4e-5/7,5e-6) ze `run-mixed-long`.

| Checkpoint | Benchmark 38 (stare klasy) | Benchmark 82 (nowe klasy) |
|---|---|---|
| `run-mixed-long` (v0.2) | 31/38 (81,6%), F1 0,913 | 2/82 (2,4%), F1 0,482 |
| `run-v3` (500 kroków) | 33/38 (86,8%), F1 0,932 | 14/82 (17,1%), F1 0,563 |
| **`run-v3b` (v0.3)** | **34/38 (89,5%)**, F1 0,945 | **40/82 (48,8%)**, F1 0,731 |

Nowe klasy po v0.3: jabłko 4/4, kot 4/4, parasol 4/4, auto 3/4, nóż 3/4,
piekarnik 3/3, pizza 3/3, łyżka 3/4, pomarańcza 2/3, zlew 2/3; wciąż nie
nauczone: banan 0/3, pies 0/4, pluszowy miś 0/4, lodówka 0/3, marchewka 0/3.
**82 zdjęcia to nadal mała próba; 48,8% to nie jest skuteczność uliczna.**
Dalszy trening potrzebny — to samo zdanie co po pierwszej iteracji.

**Czytanie cyfr przez model** (stary zbiór, sceny kwota/data/cena, 414 cyfr):
65,5% → **71,7%** trafień pozycyjnych; data ważności 5→8/24 exact, cena 1→3/24,
kwoty na rachunkach nadal 0/24 (model ich nie czyta — w aplikacji robi to OCR).
Zdania syntetyczne bez regresji: 35/122 exact, F1 0,830 (v0.2: 34/122, 0,829).

### Trzecia iteracja — v0.4: sceny wnętrz i roślin (2026-09-25)

**Test rzeczywisty użytkownika:** korytarz/przedpokój nierozpoznany, kwiaty na
szafce nazwane „Szafkak.” — model miał zamknięty słownik 40 klas obiektów i żadnych
scen wnętrz; „roślina w doniczce” była 0/2 w benchmarku.

**Dane:** 11 nowych scen (`wnetrza/*`, `rosliny/*` w `scenes.py`, katalog 200 → 211):
korytarz, przedpokój, pokój, salon, kuchnia, łazienka, schody, kwiaty w doniczce,
roślina doniczkowa, kwiaty w wazonie, kwiaty na szafce. Zdjęcia z Commons pobrane
zapytaniami angielskimi (polskie „salon”/„pokój”/„kuchnia” trafiały w nazwy miejsc
i kanał TV — `fetch_commons._queries` kładzie teraz hasło angielskie najpierw).
Recenzja ręczna **51/88** (`photos/approved-scenes.txt`): odrzucono malarstwo
(Pieter de Hooch, Rousseau, Brueghel), jaskinie zamiast schodów, bukiety bez wazonu.
Trening: sceny ×20 w miksie (2771 próbek), 1000 kroków LR 4e-5 + 800 kroków LR 8e-5.

| Benchmark | v0.3 (`run-v3b`) | **v0.4 (`run-v4b`)** |
|---|---|---|
| Sceny wnętrz (11 zdjęć) | 0/11 (0%), F1 0,426 | **6/11 (54,5%)**, F1 0,742 |
| Nowe klasy obiektów (82) | 40/82 (48,8%) | **63/82 (76,8%)**, F1 0,882 |
| Stare klasy obiektów (38) | 34/38 (89,5%) | 32/38 (84,2%), F1 0,918 |
| Czytanie cyfr (414) | 71,7% | **75,1%** |

Sceny po v0.4: korytarz ✓, pokój ✓, kuchnia ✓, łazienka ✓, schody ✓, roślina
doniczkowa ✓; słabe: salon → „kanapa”, kwiaty w doniczce → „kwiateczka w doniczkowa”,
kwiaty w wazonie → „wazon” (nazwa pojemnika zamiast zawartości). Przedpokój → „To
korytarz.” — semantycznie blisko, ale nie exact. **11 zdjęć scen to minimalna próba.**

Zapis uwag: przy pierwszym przebiegu (sceny 5× w miksie, LR 4e-5) model w ogóle
nie nauczył nowych słów (0/11, wyniki identyczne z baseline) — klasa scenowa
potrzebuje albo dużo więcej zdjęć, albo wyższego LR i ~20× nadreprezentacji.
To samo dotyczy przyszłych klas: sama obecność w miksie nie wystarczy.

### Czwarta iteracja — v0.5: tubki, polskie znaki, branding (2026-09-25)

**Zgłoszenia użytkownika (po teście na telefonie):** „tubkę z maścią nazywa
butelką”, komunikat startowy zniekształcony („Model gotowo”), zbędne „to może
być” przy obiektach, polskie znaki w tekście OCR.

**Poprawki aplikacji** (`app/`):
- komunikat startowy: „Model gotowy. Small LLM by Fabryka AI. Możesz robić zdjęcia.”;
- `model_router.dart` — usunięty prefiks „Może to być: ” z odpowiedzi SLAYER;
- `ocr_answers.dart` — leksykon `_polishWords` przywraca ogonki w wyrazach
  zwracanych przez ML Kit („waznosc” → „Ważność”; nieznane wyrazy bez zmian),
  a napis zasilania pokazywany w oryginalnej pisowni ze zdjęcia („WŁĄCZ”, nie
  „WLACZ”). Testy: 6/6 (`flutter test`), `flutter analyze` — 0 problemów.

**Dane modelu:** 4 sceny opakowań leków (`leki/masc` „To tubka maści.”,
`leki/krople`, `leki/syrop`, `rozne/pasta`), 13 zatwierdzonych zdjęć po recenzji
(32 pobrane; odrzucono lampy próżniowe zamiast tubek, probówki, reklamy,
„gel douche”). Zdjęcia scen ×22 w miksie, 800 kroków LR 4e-5/1e-5.

| Benchmark | v0.4 (`run-v4b`) | **v0.5 (`run-v6`)** |
|---|---|---|
| Sceny+leki (15 zdjęć) | — (8/15 na v5: „tubka maści” = „butelka”) | **8/15** — „To tubka maści.” ✓, krople ✓ |
| Obiekty nowe klasy (82) | 63/82 (76,8%) | **66/82 (80,5%)**, F1 0,901 |
| Obiekty stare klasy (38) | 32/38 (84,2%) | 30/38 (78,9%) — kanapy uciekły w „salon/pokój” |

Naprawione dokładnie zgłoszenie „tubka maści → butelka”. Słabe: tubka pasty
(jeszcze „butelka”), syrop („łyżka” — łyżka na zdjęciu dominuje), salon,
przedpokój (→ „korytarz”). Koszt: 2 obiekty ze starego benchmarku.

Przepis iteracji 2:

```bash
python -m slayer_vision.build_dataset --out out/dataset2 --per-kind 80 --seed 43
python -m slayer_vision.build_coco_objects \
    --annotations-zip out/annotations_trainval2017-secure.zip --out out/coco-objects2 \
    --train-per-class 32 --eval-per-class 4 --reviewed-val coco_val_reviewed2.txt
python -m slayer_vision.merge_caption_sets --out out/mixed2/train.jsonl \
    out/dataset2/train.jsonl out/dataset2/images \
    out/coco-objects2/train.jsonl out/coco-objects2/images
python -m slayer_vision.train --data-jsonl out/mixed2/train.jsonl \
    --resume-from out/run-mixed-long --reset-optimizer --output-dir out/run-v3 \
    --steps 500 --batch-size 1 --lr-projector 8e-5 --lr-lora 1.5e-5 --save-every 100
python -m slayer_vision.train --data-jsonl out/mixed2/train.jsonl \
    --resume-from out/run-v3 --reset-optimizer --output-dir out/run-v3b \
    --steps 1000 --batch-size 1 --lr-projector 4e-5 --lr-lora 7.5e-6 --save-every 250
```

Przepis iteracji 1 (`run-mixed-long`, checkpoint wydania v0.2):

```bash
python -m slayer_vision.merge_caption_sets --out out/mixed/train.jsonl \
    out/dataset/train-synthetic.jsonl out/dataset/images \
    out/coco-objects/train-target-classes.jsonl out/coco-objects/images
python -m slayer_vision.train --data-jsonl out/mixed/train.jsonl \
    --resume-from out/run-final --reset-optimizer --output-dir out/run-coco-objects \
    --steps 60 --batch-size 1 --lr-projector 1.5e-4 --lr-lora 3e-5 --save-every 20
python -m slayer_vision.train --data-jsonl out/mixed/train.jsonl \
    --resume-from out/run-coco-objects --reset-optimizer --output-dir out/run-mixed \
    --steps 120 --batch-size 1 --lr-projector 1.5e-4 --lr-lora 3e-5 --save-every 40
python -m slayer_vision.train --data-jsonl out/mixed/train.jsonl \
    --resume-from out/run-mixed --reset-optimizer --output-dir out/run-mixed-long \
    --steps 300 --batch-size 1 --lr-projector 8e-5 --lr-lora 1.5e-5 --save-every 100
```

## Struktura

```
app/                      # aplikacja Flutter (android + ios)
  lib/
    main.dart             # start, brama „model gotowy?" → Home / Setup
    src/
      prompts.dart     # predefiniowane pytania PL, komunikaty statusu
      screens/
        home_screen.dart  # kamera + duży przycisk + odpowiedź
        setup_screen.dart # pobranie/wskazanie modelu
      services/
        gemma_service.dart# Gemma 3n on-device: instalacja, aktywacja, ask()
        tts_service.dart  # synteza mowy pl-PL
        stt_service.dart  # rozpoznawanie mowy pl-PL
```

## Instalacja na Androidzie

**Pliki do pobrania:** [GitHub Releases — v0.5.0-objects](https://github.com/PiotrStyla/Small_LLM_implementation/releases/tag/v0.5.0-objects):

| Plik | Rozmiar | Zawartość |
|---|---:|---|
| `Asystent-wzrokowy.apk` | 224 513 514 B (~214 MiB) | aplikacja Flutter arm64 z OCR offline; podpisana **kluczem debugowym**. **Nowy APK** — poprawki: branding „Small LLM by Fabryka AI” w komunikacie startowym, bez „to może być”, przywracanie polskich znaków w OCR |
| `SLAYER-Vision-ONNX-IR9-v0.5.zip` | 898 088 895 B (~856 MiB) | folder `slayer-model/` z plikami modelu v0.5 (40 klas + sceny wnętrz + opakowania leków), `MODEL-ATTRIBUTION.txt` i `coco-attribution.jsonl` |

Suma SHA-256 APK: `1e8a805b885798d4031c690fafd3d93f29b4f34b19acfdf2469461679dc672fb`  
Suma SHA-256 ZIP: `5870ff36cc51af47339792fe6271df4b2d61b23d63c596c84e97562fa0ec17d4`

**Wymagania:** Android arm64, kilka GB wolnej pamięci wewnętrznej i dużo RAM;
telefon Samsung SM-A226B przy próbach zgłaszał zamknięcia `LOW_MEMORY`.
Na komputerze: [Android Platform Tools (ADB)](https://developer.android.com/tools/releases/platform-tools),
na telefonie włączone debugowanie USB i zaakceptowane uprawnienie ADB.
Pobierz **oba** pliki do tego samego folderu. W PowerShell uruchom:

```powershell
Get-FileHash .\Asystent-wzrokowy.apk -Algorithm SHA256
Get-FileHash .\SLAYER-Vision-ONNX-IR9.zip -Algorithm SHA256
adb devices
adb install -r .\Asystent-wzrokowy.apk
Expand-Archive .\SLAYER-Vision-ONNX-IR9.zip -DestinationPath .
adb shell mkdir -p /sdcard/Android/data/ai.slayer.vision_assistant/files/slayer-model
adb push slayer-model/. /sdcard/Android/data/ai.slayer.vision_assistant/files/slayer-model/
adb shell am force-stop ai.slayer.vision_assistant
adb shell am start -n ai.slayer.vision_assistant/.MainActivity
```

`adb devices` musi pokazać telefon jako `device`, nie `unauthorized`.
Sprawdź sumy z tabelą przed instalacją. Pliki `.onnx.data` muszą leżeć **obok**
odpowiadających im `.onnx`; nie uruchamiaj ONNX bez całego folderu. Po
uruchomieniu aplikacja szuka katalogu `slayer-model` i wybiera SLAYER, jeśli
jest obecny. Gdy widzisz ekran konfiguracji, sprawdź pliki przez
`adb shell ls /sdcard/Android/data/ai.slayer.vision_assistant/files/slayer-model`.
Model można też pobrać z [Hugging Face](https://huggingface.co/PiotrSty/slayer-vision-onnx)
(te same osiem wymaganych plików, bez tokenu). Wskazanie folderu przez
systemowy picker Androida nie zostało zweryfikowane; ADB to sprawdzona ścieżka.

**Bez komputera:** samo APK zainstaluje aplikację, ale podpisy zdjęć SLAYER
nie działają bez osobnego modelu. Na ekranie konfiguracji można zamiast
SLAYER wybrać Gemma 3n E2B (`google/gemma-3n-E2B-it-litert-lm`, ~2 GB):
zaakceptuj licencję na Hugging Face i podaj własny token `hf_…` do pobrania
lub wskaż posiadany plik `.litertlm`. Jeśli folder SLAYER jest zainstalowany,
ma pierwszeństwo przy starcie. Gemma na tym Samsungu nie była zweryfikowana.

### Budowanie ze źródeł

Flutter ≥ 3.44 (lokalnie 3.47.5). `cd app`, `flutter pub get`,
`flutter build apk --release`. APK release jest na razie podpisywany kluczem
debugowym; przed publicznym sklepem trzeba go zastąpić kluczem wydawcy.
Wersja iOS wymaga iOS 15.5+, macOS/Xcode i podpisania aplikacji; **IPA nie
zostało zbudowane ani przetestowane**. W tym wydaniu pliki instalacyjne są
wyłącznie dla Androida.

## Rzeczywiste niedomagania i wyniki prób

| Obszar | Zaobserwowano / granica możliwości |
|---|---|
| Podpis zdjęcia | Model v0.5: obiekty 66/82 (80,5%) i 30/38 (78,9%), sceny+opakowania leków 8/15 — „To tubka maści.” ✓ (naprawione zgłoszenie „tubka → butelka”), krople ✓, korytarz, pokój, kuchnia, łazienka; słabe: tubka pasty („butelka”), syrop („łyżka”), salon, przedpokój (→ „korytarz”), kanapy bywają nazywane „salonem”. Na zdjęciach realnych bywa ucinany kwalifikator („To kostka masła.” → „To kostka.”). Historyczny wynik 114/319 na dawnym `eval.jsonl` dotyczy zbioru z **błędnymi etykietami** — nie jest miarą jakości. Czas odpowiedzi bywa liczony w dziesiątkach sekund. Brak wiarygodnego wskaźnika pewności i filtra rozmazania. |
| Polecenia | SLAYER nie jest VQA i nie potrafi rozumieć swobodnych pytań. „Co jest przede mną?”/„Co to jest?” to ten sam podpis zdjęcia. Tekst/data/kwota idą przez osobny OCR. „Który przycisk?” identyfikuje wyłącznie czytelny napis zasilania; nie wskazuje położenia ani ikony. |
| OCR | ML Kit Latin działa w release na Samsungu, lecz odręczny numer `12-266-85-28` odczytał jako `R-G6-8S-28`. Reguły dat/rachunków ograniczają zgadywanie, lecz nie naprawiają błędnie odczytanej cyfry. ML Kit zwraca polskie znaki niekonsekwentnie (bez ogonków) — aplikacja przywraca je wyłącznie dla wyrazów ze swojego leksykonu (`_polishWords` w `ocr_answers.dart`), nieznane wyrazy zostają jak z OCR. Bez etykiety terminu/kwoty aplikacja odmawia odpowiedzi. **Nie używać samego OCR do decyzji o leku, terminie lub zapłacie.** |
| Pamięć | Trzy grafy fp32 to ~897 MB na dysku; sesje ONNX na Samsungu osiągały ~0,9–1,1 GB PSS i system zapisał kilka zamknięć `LOW_MEMORY`. Nie ma kwantyzacji ani odciążenia sesji po podpisie. Stabilność długotrwała i działanie na słabszych urządzeniach nie są potwierdzone. |
| Głos / platformy | TTS jest wywoływane w kodzie, lecz fizyczny odsłuch nie został potwierdzony. Systemowy ASR może potrzebować sieci; przy pytaniu poza katalogiem SLAYER odmawia. iOS/IPA i obsługa z czytnikiem ekranu nie były testowane. |

Ta wersja jest **prototypem badawczym, nie narzędziem bezpieczeństwa**.

## Licencje

Kod aplikacji: MIT (`LICENSE`). **Wagi modelu są osobnym utworem**:
baza językowa goLLeM (Arkadiusz Słota / SlayerLab) —
[CC-BY-SA-4.0](https://huggingface.co/SlayerLab/goLLeM-110M-PL-SFT-merged);
enkoder SigLIP Google —
[Apache-2.0](https://huggingface.co/google/siglip-base-patch16-224).
Archiwum modelu podaje autorów, pochodzenie i zmiany w
`MODEL-ATTRIBUTION.txt`. Licencja MIT repozytorium **nie obejmuje** wag.
