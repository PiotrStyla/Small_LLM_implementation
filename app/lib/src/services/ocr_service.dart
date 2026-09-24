import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

import '../prompts.dart';
import 'ocr_answers.dart';

/// Odczytuje pełnej rozdzielczości JPEG z kamery, nie miniaturę 224×224
/// używaną przez SLAYER. ML Kit Latin jest dołączony do aplikacji offline.
abstract final class OcrService {
  static Future<String> answer(String imagePath, AskIntent intent) async {
    final recognizer = TextRecognizer(script: TextRecognitionScript.latin);
    try {
      final text = await recognizer.processImage(
        InputImage.fromFilePath(imagePath),
      );
      return answerFromOcr(intent, text.text.split('\n'));
    } finally {
      await recognizer.close();
    }
  }
}
