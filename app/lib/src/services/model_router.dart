import 'dart:typed_data';

import 'gemma_service.dart';
import 'slayer_service.dart';

/// Wybór silnika odpowiedzi: Gemma 3n (flutter_gemma) albo SLAYER-Vision
/// (własny goLLeM w ONNX). Ustawiany przy wczytywaniu modelu w Setup.
enum EngineKind { gemma, slayer }

abstract final class ModelRouter {
  static EngineKind engine = EngineKind.gemma;

  static bool get slayerReady => SlayerService.instance.isReady;

  /// Strumień tokenów odpowiedzi dla zdjęcia.
  static Stream<String> ask(Uint8List jpeg, String question) {
    switch (engine) {
      case EngineKind.slayer:
        return SlayerService.instance.ask(jpeg);
      case EngineKind.gemma:
        return GemmaService.instance.ask(jpeg, question);
    }
  }
}
