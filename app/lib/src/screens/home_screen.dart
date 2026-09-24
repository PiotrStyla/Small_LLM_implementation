import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../prompts.dart';
import '../services/model_router.dart';
import '../services/stt_service.dart';
import '../services/tts_service.dart';

/// Ekran główny: podgląd z kamery, jeden duży przycisk „Pokaż i zapytaj",
/// predefiniowane pytania i sterowanie głosem.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _camera;
  bool _busy = false;
  String _status = '';
  String _answer = '';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _camera?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _camera;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      controller.dispose();
      _camera = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        _speakError(StatusTexts.noCamera);
        return;
      }
      // Tylny aparat: użytkownik kieruje telefon na przedmioty przed sobą.
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _camera = controller);
    } catch (e) {
      _speakError(StatusTexts.noCamera);
    }
  }

  Future<void> _speakError(String message) async {
    setState(() => _status = message);
    await TtsService.instance.speak(message);
  }

  /// Zdjęcie → opis sceny lub zadanie OCR → odpowiedź głosowa.
  Future<void> _ask(String question, {AskIntent? intent}) async {
    final controller = _camera;
    if (_busy || controller == null || !controller.value.isInitialized) {
      if (controller == null) await _speakError(StatusTexts.noCamera);
      return;
    }
    setState(() {
      _busy = true;
      _answer = '';
      _status = StatusTexts.capturing;
    });
    await TtsService.instance.speak(StatusTexts.capturing);
    try {
      final XFile shot = await controller.takePicture();
      // OCR czyta JPEG ze ścieżki w pełnej rozdzielczości; nie kopiuj zdjęcia
      // do pamięci Dart, gdy nie uruchamiamy modelu obrazowego.
      final Uint8List? jpeg =
          ModelRouter.engine == EngineKind.gemma ||
              intent == AskIntent.scene ||
              intent == AskIntent.object
          ? await shot.readAsBytes()
          : null;
      setState(() => _status = StatusTexts.thinking);
      await TtsService.instance.speak(StatusTexts.thinking);

      final buffer = StringBuffer();
      await for (final token in ModelRouter.ask(
        jpeg,
        question,
        imagePath: shot.path,
        intent: intent,
      )) {
        buffer.write(token);
        if (mounted) setState(() => _answer = buffer.toString());
      }

      final answer = buffer.toString().trim();
      if (answer.isEmpty) {
        await _speakError(StatusTexts.noAnswer);
      } else {
        setState(() => _status = '');
        await TtsService.instance.speak(answer);
      }
    } catch (e) {
      await _speakError('${StatusTexts.noAnswer} $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _askByVoice() async {
    if (_busy) return;
    setState(() => _status = StatusTexts.listening);
    await TtsService.instance.speak(StatusTexts.listening);
    final question = await SttService.instance.listenOnce();
    if (question.isEmpty) {
      await _speakError(StatusTexts.noSpeech);
      return;
    }
    await _ask(question, intent: intentForVoice(question));
  }

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: camera == null || !camera.value.isInitialized
                  ? const Center(child: CircularProgressIndicator())
                  : Semantics(
                      label: 'Podgląd z tylnej kamery',
                      child: CameraPreview(camera),
                    ),
            ),
            if (_status.isNotEmpty || _answer.isNotEmpty)
              _AnswerPanel(status: _status, answer: _answer, busy: _busy),
            SizedBox(
              height: 76,
              child: ListView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                children: [
                  for (final preset in kPresets)
                    Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: Semantics(
                        button: true,
                        label: 'Zapytaj: ${preset.label}',
                        hint: preset.intent == AskIntent.powerButton &&
                                ModelRouter.engine == EngineKind.slayer
                            ? 'W trybie SLAYER rozpoznaje wyłącznie czytelny napis zasilania; nie wskazuje położenia ani samej ikony.'
                            : null,
                        child: ActionChip(
                          avatar: Icon(preset.icon, size: 28),
                          label: Text(
                            preset.intent == AskIntent.powerButton &&
                                    ModelRouter.engine == EngineKind.slayer
                                ? 'Który przycisk? (napis)'
                                : preset.label,
                            style: const TextStyle(fontSize: 18),
                          ),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 12,
                          ),
                          onPressed: _busy
                              ? null
                              : () => _ask(
                                  preset.question,
                                  intent: preset.intent,
                                ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 12),
              child: Row(
                children: [
                  Expanded(
                    child: Semantics(
                      button: true,
                      label: StatusTexts.askButton,
                      hint: StatusTexts.askButtonHint,
                      child: SizedBox(
                        height: 110,
                        child: FilledButton.icon(
                          style: FilledButton.styleFrom(
                            textStyle: const TextStyle(
                              fontSize: 26,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          onPressed: _busy
                              ? null
                              : () => _ask(
                                  kPresets.first.question,
                                  intent: kPresets.first.intent,
                                ),
                          icon: const Icon(Icons.camera_alt, size: 40),
                          label: const Text(StatusTexts.askButton),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Semantics(
                    button: true,
                    label: StatusTexts.micButton,
                    child: SizedBox(
                      height: 110,
                      width: 110,
                      child: FilledButton.tonalIcon(
                        onPressed: _busy ? null : _askByVoice,
                        icon: const Icon(Icons.mic, size: 40),
                        label: const Text(''),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _AnswerPanel extends StatelessWidget {
  const _AnswerPanel({
    required this.status,
    required this.answer,
    required this.busy,
  });

  final String status;
  final String answer;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(maxHeight: 180),
      margin: const EdgeInsets.symmetric(horizontal: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey.shade900,
        borderRadius: BorderRadius.circular(16),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (status.isNotEmpty)
              Row(
                children: [
                  if (busy)
                    const Padding(
                      padding: EdgeInsets.only(right: 10),
                      child: SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
                  Text(
                    status,
                    style: const TextStyle(fontSize: 18, color: Colors.white70),
                  ),
                ],
              ),
            if (answer.isNotEmpty)
              Text(
                answer,
                style: const TextStyle(
                  fontSize: 24,
                  color: Colors.white,
                  height: 1.3,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
