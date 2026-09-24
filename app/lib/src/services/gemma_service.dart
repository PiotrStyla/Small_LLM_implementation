import 'dart:typed_data';

import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';

/// Most do Gemma 3n działającej lokalnie (LiteRT-LM, format `.litertlm`).
///
/// Świadomie jeden silnik i jeden format: `.litertlm` obsługuje oba systemy
/// (Android arm64, iOS 15+) i multimodalność (obraz) bez MediaPipe.
class GemmaService {
  GemmaService._();

  static final GemmaService instance = GemmaService._();

  /// Domyślne repozytorium: Gemma 3n E2B (multimodalny, ~2 GB, on-device).
  static const String defaultModelRepo = 'google/gemma-3n-E2B-it-litert-lm';

  /// Kontekst sesji: klatka JPEG (~257 tokenów) + pytanie + krótka odpowiedź.
  static const int _maxTokens = 2048;

  InferenceModel? _model;
  ModelRuntimeDefaults? _defaults;

  bool get isReady => _model != null;

  /// Czy model jest już zainstalowany w pamięci urządzenia
  /// (przywracane automatycznie po restarcie aplikacji).
  bool get hasModel => FlutterGemma.hasActiveModel();

  /// Inicjalizacja SDK — raz na start aplikacji.
  Future<void> init({String? huggingFaceToken}) {
    return FlutterGemma.initialize(
      huggingFaceToken: huggingFaceToken,
      inferenceEngines: [LiteRtLmEngine()],
    );
  }

  /// Pobiera i instaluje model z Hugging Face (manifest repozytorium wskazuje
  /// wariant dla urządzenia). Wymaga tokena, gdy repo jest ograniczone licencją.
  Future<void> installFromHuggingFace({
    required String repo,
    String? token,
    void Function(int progress)? onProgress,
  }) async {
    final builder = FlutterGemma.installModel(
      modelType: ModelType.gemmaIt,
      fileType: ModelFileType.litertlm,
    ).fromHuggingFace(repo, token: token);
    if (onProgress != null) {
      builder.withProgress(onProgress);
    }
    final installation = await builder.install();
    _defaults = installation.runtime;
    await activate();
  }

  /// Instaluje model z pliku `.litertlm` wskazanego przez użytkownika.
  Future<void> installFromFile(String path) async {
    final installation = await FlutterGemma.installModel(
      modelType: ModelType.gemmaIt,
      fileType: ModelFileType.litertlm,
    )
        .fromFile(path)
        .install();
    _defaults = installation.runtime;
    await activate();
  }

  /// Ładuje aktywny model do pamięci z obsługą obrazu.
  Future<void> activate() async {
    _model = await FlutterGemma.getActiveModel(
      defaults: _defaults,
      maxTokens: _maxTokens,
      supportImage: true,
      maxNumImages: 1,
    );
  }

  /// Jedno zapytanie: klatka + pytanie → strumień tokenów odpowiedzi po polsku.
  ///
  /// Każde zapytanie dostaje świeżą sesję — poprzednie zdjęcie nie może
  /// wpływać na kolejną odpowiedź.
  Stream<String> ask(Uint8List imageJpeg, String question) async* {
    final model = _model;
    if (model == null) {
      throw StateError('Model nie jest gotowy. Uruchom install/activate.');
    }
    final chat = InferenceChat(
      sessionCreator: () => model.createSession(),
      maxTokens: _maxTokens,
      supportImage: true,
    );
    try {
      await chat.addQuery(
        Message.withImage(text: question, imageBytes: imageJpeg, isUser: true),
      );
      await for (final response in chat.generateChatResponseAsync()) {
        if (response is TextResponse) {
          yield response.token;
        }
      }
    } finally {
      // Sesja natywna trzyma kontekst (w tym obraz) — zamykamy po każdym
      // zapytaniu, żeby nie obciążać pamięci między „pokaż i zapytaj".
      await chat.close();
    }
  }
}
