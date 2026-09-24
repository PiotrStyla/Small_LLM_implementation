import 'dart:async';

import 'package:speech_to_text/speech_to_text.dart';

/// Rozpoznawanie mowy (systemowy ASR) — jedno pytanie na wciśnięcie mikrofonu.
class SttService {
  SttService._();

  static final SttService instance = SttService._();

  final SpeechToText _stt = SpeechToText();
  bool _available = false;

  Future<bool> init() async {
    if (_available) return true;
    _available = await _stt.initialize();
    return _available;
  }

  bool get isAvailable => _available;

  /// Nasłuchuje jednego pytania i zwraca rozpoznane słowa.
  ///
  /// Zwraca pusty ciąg, gdy ASR nie jest dostępny lub nie rozpoznał mowy.
  Future<String> listenOnce() async {
    if (!await init()) return '';
    final completer = Completer<String>();
    await _stt.listen(
      onResult: (result) {
        if (result.finalResult && !completer.isCompleted) {
          completer.complete(result.recognizedWords.trim());
        }
      },
      listenOptions: SpeechListenOptions(
        localeId: 'pl_PL',
        cancelOnError: true,
      ),
    );
    return completer.future;
  }

  Future<void> stop() => _stt.stop();
}
