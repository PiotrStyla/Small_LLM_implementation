import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

/// SLAYER-Vision (goLLeM-110M-PL-SFT + SigLIP) w grafach ONNX — własny,
/// polski model na urządzeniu, alternatywa dla Gemma 3n.
///
/// Współpracuje z eksportem `slayer_vision.export_onnx`:
/// `vision_projector.onnx`, `lm_embeds.onnx`, `embed_tokens.onnx`,
/// `tokens_decoded.json` w jednym katalogu (wagi w `*.onnx.data` obok grafów).
///
/// Dekodowanie greedy jest identyczne jak w teście parzystości eksportu:
/// tokeny obrazu (196) + embeddy wygenerowanych tokenów → logits → argmax.
/// Format treningowy nie ma BOS/promptu — model generuje samo zdanie.
class SlayerService {
  SlayerService._();

  static final SlayerService instance = SlayerService._();

  static const int imageSize = 224;
  static const int imageTokens = 196;
  static const int hidden = 768;
  static const int eosId = 0; // eos_token_id tokenizera GoLLeM
  static const int maxNewTokens = 32;

  OrtSession? _vision;
  OrtSession? _lm;
  OrtSession? _embed;
  List<String> _tokens = const [];

  bool get isReady =>
      _vision != null && _lm != null && _embed != null && _tokens.isNotEmpty;

  /// Wczytuje komplet plików modelu z katalogu.
  Future<void> loadFromDirectory(String dirPath) async {
    OrtEnv.instance.init();
    final dir = Directory(dirPath);
    Future<OrtSession> session(String name) async {
      final bytes = await File('${dir.path}${Platform.pathSeparator}$name').readAsBytes();
      return OrtSession.fromBuffer(bytes, OrtSessionOptions());
    }

    final tokensFile = File('${dir.path}${Platform.pathSeparator}tokens_decoded.json');
    if (!await tokensFile.exists()) {
      throw StateError('Brak tokens_decoded.json w $dirPath');
    }
    _vision = await session('vision_projector.onnx');
    _lm = await session('lm_embeds.onnx');
    _embed = await session('embed_tokens.onnx');
    _tokens = (jsonDecode(await tokensFile.readAsString()) as List).cast<String>();
  }

  /// JPEG → tensor wejściowy vision: (1,3,224,224), piksele w [-1,1].
  Float32List _preprocess(Uint8List jpeg) {
    final decoded = img.decodeImage(jpeg);
    if (decoded == null) {
      throw ArgumentError('Nie udało się zdekodować JPEG');
    }
    final resized = img.copyResize(
      decoded,
      width: imageSize,
      height: imageSize,
      interpolation: img.Interpolation.linear,
    );
    final out = Float32List(3 * imageSize * imageSize);
    var index = 0;
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < imageSize; y++) {
        for (var x = 0; x < imageSize; x++) {
          final pixel = resized.getPixel(x, y);
          final value = switch (c) { 0 => pixel.r, 1 => pixel.g, _ => pixel.b };
          out[index++] = value / 255.0 * 2.0 - 1.0;
        }
      }
    }
    return out;
  }

  List<double> _flatten(dynamic value) {
    if (value is List<double>) return value;
    if (value is List) {
      final out = <double>[];
      for (final element in value) {
        out.addAll(_flatten(element));
      }
      return out;
    }
    throw StateError('Nieoczekiwany typ tensora: ${value.runtimeType}');
  }

  Future<List<double>> _runFloat(
    OrtSession session,
    Map<String, OrtValueTensor> inputs,
  ) async {
    final runOptions = OrtRunOptions();
    try {
      final outputs = await session.runAsync(runOptions, inputs);
      final value = _flatten(outputs?[0]?.value);
      outputs?.forEach((o) => o?.release());
      return value;
    } finally {
      runOptions.release();
      for (final tensor in inputs.values) {
        tensor.release();
      }
    }
  }

  /// Zdanie dla zdjęcia — strumień kolejnych fragmentów tekstu.
  Stream<String> ask(Uint8List jpeg) async* {
    final vision = _vision;
    final lm = _lm;
    final embed = _embed;
    if (vision == null || lm == null || embed == null || _tokens.isEmpty) {
      throw StateError('Model SLAYER nie jest wczytany');
    }

    final imageEmbeds = await _runFloat(vision, {
      'pixel_values': OrtValueTensor.createTensorWithDataList(
        _preprocess(jpeg),
        [1, 3, imageSize, imageSize],
      ),
    });

    final ids = <int>[];
    for (var step = 0; step < maxNewTokens; step++) {
      final textEmbeds = ids.isEmpty
          ? <double>[]
          : await _runFloat(embed, {
              'ids': OrtValueTensor.createTensorWithDataList(
                Int64List.fromList(ids),
                [1, ids.length],
              ),
            });

      final seq = imageTokens + ids.length;
      final embeds = Float32List(seq * hidden)
        ..setAll(0, imageEmbeds)
        ..setAll(imageTokens * hidden, textEmbeds);
      final logits = await _runFloat(lm, {
        'inputs_embeds':
            OrtValueTensor.createTensorWithDataList(embeds, [1, seq, hidden]),
        'attention_mask': OrtValueTensor.createTensorWithDataList(
          Int64List(seq),
          [1, seq],
        ),
      });

      // Argmax po ostatniej pozycji — logits mają (1, seq, vocab).
      final vocab = _tokens.length;
      final offset = (seq - 1) * vocab;
      var bestId = 0;
      var bestLogit = -double.infinity;
      for (var i = 0; i < vocab; i++) {
        final logit = logits[offset + i];
        if (logit > bestLogit) {
          bestLogit = logit;
          bestId = i;
        }
      }
      if (bestId == eosId) break;
      ids.add(bestId);
      yield _tokens[bestId];
    }
  }
}
