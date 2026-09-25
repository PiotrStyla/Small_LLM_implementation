import 'dart:typed_data';

import '../prompts.dart';

import 'gemma_service.dart';
import 'ocr_service.dart';
import 'slayer_service.dart';

/// Gemma rozumie pytanie + zdjęcie. SLAYER tylko podpisuje obraz; pytania
/// o tekst, datę, kwotę i opisany napisami panel obsługuje lokalny OCR.
enum EngineKind { gemma, slayer }

abstract final class ModelRouter {
  static EngineKind engine = EngineKind.gemma;

  static bool get slayerReady => SlayerService.instance.isReady;

  /// Pytanie jest przekazywane Gemmie; SLAYER używa jawnego rodzaju zadania.
  static Stream<String> ask(
    Uint8List? jpeg,
    String question, {
    required String imagePath,
    required AskIntent? intent,
  }) async* {
    if (engine == EngineKind.gemma) {
      yield* GemmaService.instance.ask(jpeg!, question);
      return;
    }
    switch (intent) {
      case AskIntent.scene:
      case AskIntent.object:
        yield* SlayerService.instance.ask(jpeg!);
        return;
      case AskIntent.readText:
      case AskIntent.expiry:
      case AskIntent.payment:
      case AskIntent.powerButton:
        yield await OcrService.answer(imagePath, intent!);
        return;
      case null:
        yield 'Model SLAYER rozpoznaje tylko dostępne polecenia. Wybierz opis zdjęcia, tekst, datę, kwotę albo przycisk z czytelnym napisem.';
    }
  }
}
