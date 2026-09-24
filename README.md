# Small LLM implementation — Asystent wzrokowy (on-device)

Asystent dla osób niewidomych i seniorów: **pokaż i zapytaj**. Telefon opisuje
to, co widać w kamerze — po polsku, głosem, w całości na urządzeniu (Android
i iOS), bez serwera inference i bez wysyłania zdjęć do internetu.

```
kamera → klatka JPEG → Gemma 3n (lokalnie) → krótkie zdanie po polsku → TTS → 🔊
```

Projekt realizuje „przypadek 5" z analizy: mały model multimodalny na
urządzeniu jako osobisty asystent świata fizycznego (Gemma Vision, Vite Vere
Offline, Guided Vision jako punkty odniesienia).

## Możliwości (MVP)

- 📸 **Jeden duży przycisk „Pokaż i zapytaj"** — zdjęcie + domyślne pytanie
  „Co jest przede mną?".
- 💊 **Predefiniowane pytania**: „Co to jest?", „Przeczytaj tekst",
  „Data ważności", „Ile do zapłaty?", „Który przycisk?".
- 🎤 **Pytania głosem** (systemowy ASR, polski).
- 🔊 **Odpowiedź głosowa** (systemowy TTS — ten sam głos/tempo, którego
  użytkownik używa w czytniku ekranu) + duży, czytelny tekst.
- 📴 **Offline po pobraniu modelu** — obrazy nie opuszczają urządzenia.
- ♿ **Dostępność**: pełne etykiety Semantics dla TalkBack/VoiceOver,
  przyciski min. 72 dp, sterowanie jedną ręką.

## Stack

| Warstwa | Wybór | Dlaczego |
|---|---|---|
| App | Flutter (Android + iOS, jeden kod) | flutter_gemma jest natywnie multiplatformowy |
| Model | **Gemma 3n E2B** (`.litertlm`, LiteRT-LM) | multimodalny (obraz+tekst), ~2 GB RAM, wizja na obu systemach |
| Ingerencja | `flutter_gemma` + `flutter_gemma_litertlm` | jeden format `.litertlm` na Androida i iOS (15+) |
| TTS / ASR | `flutter_tts` / `speech_to_text` | systemowe — zero dodatkowych modeli, znajomy głos |
| Kamera | `camera` | standard Fluttera |

Alternatywa badawcza (osobny tor): **SLAYER-Vision-PL** — własny ultramały
polski VLM (vision encoder ≈ 86M + projector 10–30M + goLLeM-110M-PL-SFT
≈ 210–230M param.) na bazie modeli [SlayerLab](https://huggingface.co/SlayerLab).

## SLAYER-Vision-PL (tor badawczy)

Ultramały polski VLM na bazie własnych modeli SlayerLab — przyszły, w pełni
własny model do tego samego produktu (obecny MVP używa Gemma 3n).

```
obraz 224×224 → SigLIP-base (zamrożony) → 196 tokenów → projector MLP
→ tokeny obrazu + tokeny tekstu → goLLeM-110M-PL-SFT (LoRA) → zdanie PL
```

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
python smoke_test.py                                  # forward/backward w realnych wymiarach
python -m slayer_vision.build_dataset --out out/dataset --per-kind 40 --photos photos
python -m slayer_vision.train --data-jsonl out/dataset/train.jsonl \
    --image-root out/dataset/images --steps 1000 --batch-size 4 --augment
python -m slayer_vision.evaluate --checkpoint out/run-150 \
    --data-jsonl out/dataset/eval.jsonl --image-root out/dataset/images
```

### Eksport on-device (ONNX)

flutter_gemma obsługuje gotowe rodziny modeli — własną architekturę (SigLIP +
projector + goLLeM) aplikacja składa sama z dwóch grafów ONNX:

```bash
python -m slayer_vision.export_onnx --checkpoint out/run-final --out out/export-slayer-vision
```

| Plik | Rola |
|---|---|
| `vision_projector.onnx` (+`.data`) | piksele → 196 tokenów obrazu (batch 1) |
| `lm_embeds.onnx` (+`.data`) | embeddy + maska → logits; **sekwencja dynamiczna** |
| `tokens_decoded.json` | mapa id → tekst (aplikacja tylko dekoduje — format treningowy nie ma promptu/BOS) |
| `manifest.json` | wymiary, pliki, wynik testu parzystości |

Weryfikacja przy eksporcie: **test dekodowania greedy PyTorch ↔ ONNX** — musi
wyjść identyczne zdanie (akceptacja produktowa; same różnice liczbowe fp32
rzędu 1e-3 to szum przekształceń grafu). Eksporter ma dwa pułapki zapisane
w kodzie: opset <18 psuje węzły `Split`, a `dynamo=False` pada na SDPA
(transformers 4.57).

Grafy to ~800 MB fp32 (wagi w `*.onnx.data`) — na telefon kwantyzacja int8
(`onnxruntime.quantization`) zwinie do ~200 MB.

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
2. **Zdjęcia** — dwa źródła:
   - **Wikimedia Commons (bez Twoich zdjęć!)** — `python -m slayer_vision.fetch_commons`
     pobiera otwarte zdjęcia (CC0/PD/CC BY/CC BY-SA) i zapisuje atrybucję
     w `photos/ATTRIBUTION.json`. Aktualnie: 692 zdjęć dla 178/200 scen.
     Trafność weryfikujesz na `photos/REVIEW.html`
     (`python -m slayer_vision.review`) — nietrafione pliki kasujesz ręcznie.
   - **Własne** — wrzucaj do `slayer_vision/photos/<scene_id>/*.jpg`;
     builder przepisuje je do zbioru ze zdaniem docelowym z katalogu.
     Lista scen bez zdjęć: `photos_missing.txt`.

```bash
python -m slayer_vision.build_dataset --out out/dataset --per-kind 40 --photos photos
```

Podział train/eval: co piąta próbka grupy → eval (nowe ujęcia tych samych
scen, nie nowe kategorie). Obecnie wygenerowane: 280 próbek syntetycznych
(224/56). Weryfikacja formatu: `CaptionDataset` + `collate` (padding maskowany
przez -100).

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

## Uruchomienie

Wymagania: Flutter **≥ 3.44** (sprawdzone na 3.47.5), Android arm64 lub
iOS 15+.

```bash
cd app
flutter pub get
flutter run
```

### Model (raz, potem offline)

Aplikacja przy pierwszym uruchomieniu prosi o model **Gemma 3n E2B**
(`google/gemma-3n-E2B-it-litert-lm`, ~2 GB):

1. Zaakceptuj licencję modelu na Hugging Face i skopiuj token `hf_…`.
2. Wklej token w ekranie konfiguracji i wybierz „Pobierz model".
3. Alternatywnie: wrzuć plik `.litertlm` na telefon i wybierz
   „Wybierz plik .litertlm" (udostępnianie plików w iOS jest włączone).

Token jest potrzebny wyłącznie do pobrania — nie jest nigdzie zapisywany
ani wysyłany dalej.

### Budowanie

```bash
flutter build apk --release        # Android (tylko arm64-v8a)
flutter build ipa --release        # iOS (wymaga macOS + certyfikatów)
```

## Ograniczenia MVP

- Model ~2 GB — pierwsze uruchomienie wymaga Wi-Fi i ~2 GB wolnego miejsca.
- Odpowiedzi są celowo krótkie i z zamkniętego katalogu pytań; otwarte
  „opisz świat" przy tej klasie modelu halucynuje — to świadomy kompromis.
- `.litertlm` = Android tylko `arm64-v8a` (zawężone w `build.gradle.kts`).

## Licencja

MIT (patrz `LICENSE`).
