import 'package:flutter_tts/flutter_tts.dart';

/// Synteza mowy (systemowy TTS urządzenia).
///
/// Systemowy TTS jest zamierzony: osoba niewidoma ma już skonfigurowany głos,
/// tempo i język czytnika ekranu — asystent używa tych samych ustawień zamiast
/// dokładać własny model syntezy.
class TtsService {
  TtsService._();

  static final TtsService instance = TtsService._();

  final FlutterTts _tts = FlutterTts();
  bool _ready = false;

  Future<void> init() async {
    await _tts.setLanguage('pl-PL');
    // Wolniejsze tempo — odpowiedzi mają być zrozumiałe dla seniorów.
    await _tts.setSpeechRate(0.45);
    await _tts.setPitch(1.0);
    await _tts.awaitSpeakCompletion(true);
    _ready = true;
  }

  /// Wypowiada [text], przerywając ewentualną wcześniejszą wypowiedź.
  Future<void> speak(String text) async {
    if (!_ready || text.trim().isEmpty) return;
    await _tts.stop();
    await _tts.speak(text);
  }

  Future<void> stop() => _tts.stop();
}
