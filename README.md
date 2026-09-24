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
python -m slayer_vision.train --data-jsonl data/captions.jsonl --image-root data/images
```

Zapis checkpointu: `projector.pt` + adaptery LoRA (`lora/`) + manifest
`slayer_vision.json` — do dołożenia w aplikacji zamiast Gemma 3n, gdy tor
treningowy domknie jakość.

**Plan danych:** specjalizowany, zamknięty katalog ~200 typów scen
(zdjęcie → jedno zdanie) — 110M model halucynuje w otwartym „opisz świat",
więc uczymy wąskiego zachowania produktowego. Źródła: synteza dokumentów
z PolOCRBench/OCR_engine (sceny „przeczytaj list/rachunek"), zdjęcia
przedmiotów codziennych i opakowań.

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
