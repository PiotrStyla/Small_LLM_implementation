import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../prompts.dart';
import '../services/gemma_service.dart';
import '../services/model_router.dart';
import '../services/slayer_service.dart';
import '../services/tts_service.dart';
import 'home_screen.dart';

/// Ekran pierwszej konfiguracji: pobranie Gemma 3n (E2B, .litertlm)
/// z Hugging Face albo wskazanie pliku modelu z pamięci urządzenia.
class SetupScreen extends StatefulWidget {
  const SetupScreen({super.key});

  @override
  State<SetupScreen> createState() => _SetupScreenState();
}

class _SetupScreenState extends State<SetupScreen> {
  final _repoController = TextEditingController(
    text: GemmaService.defaultModelRepo,
  );
  final _tokenController = TextEditingController();
  int? _progress;
  String? _error;

  @override
  void dispose() {
    _repoController.dispose();
    _tokenController.dispose();
    super.dispose();
  }

  Future<void> _run(Future<void> Function() action) async {
    setState(() {
      _error = null;
      _progress = 0;
    });
    try {
      await action();
      if (!mounted) return;
      final navigator = Navigator.of(context);
      await TtsService.instance.speak(StatusTexts.modelReady);
      navigator.pushReplacement(
        MaterialPageRoute<void>(builder: (_) => const HomeScreen()),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
      await TtsService.instance.speak('Nie udało się zainstalować modelu.');
    } finally {
      if (mounted) setState(() => _progress = null);
    }
  }

  Future<void> _installFromHub() => _run(() async {
        await GemmaService.instance.installFromHuggingFace(
          repo: _repoController.text.trim(),
          token: _tokenController.text.trim().isEmpty
              ? null
              : _tokenController.text.trim(),
          onProgress: (p) {
            if (mounted) setState(() => _progress = p);
          },
        );
      });

  Future<void> _installFromFile() async {
    final file = await FilePicker.pickFile();
    final path = file?.path;
    if (path == null) return;
    await _run(() => GemmaService.instance.installFromFile(path));
  }

  /// Wczytuje własny model SLAYER (goLLeM w ONNX) z folderu zawierającego
  /// `vision_projector.onnx`, `lm_embeds.onnx`, `embed_tokens.onnx`,
  /// `tokens_decoded.json` (oraz `*.onnx.data`).
  Future<void> _loadSlayer() async {
    final dir = await FilePicker.getDirectoryPath();
    if (dir == null) return;
    await _run(() async {
      await SlayerService.instance.loadFromDirectory(dir);
      ModelRouter.engine = EngineKind.slayer;
    });
  }

  @override
  Widget build(BuildContext context) {
    final busy = _progress != null;
    return Scaffold(
      appBar: AppBar(title: const Text('Przygotowanie asystenta')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'Asystent potrzebuje modelu Gemma 3n (E2B) w formacie .litertlm. '
            'Model działa w całości na urządzeniu — po pobraniu nie jest '
            'potrzebne połączenie z internetem.\n\n'
            'Repozytorium na Hugging Face jest ograniczone licencją — '
            'zaakceptuj ją na stronie modelu i wklej poniżej swój token.',
            style: TextStyle(fontSize: 18, height: 1.4),
          ),
          const SizedBox(height: 24),
          TextField(
            controller: _repoController,
            decoration: const InputDecoration(
              labelText: 'Repozytorium Hugging Face',
              border: OutlineInputBorder(),
            ),
            enabled: !busy,
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _tokenController,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'Token Hugging Face (hf_…)',
              border: OutlineInputBorder(),
            ),
            enabled: !busy,
          ),
          const SizedBox(height: 24),
          Semantics(
            button: true,
            label: 'Pobierz model z Hugging Face',
            child: SizedBox(
              height: 72,
              child: FilledButton.icon(
                onPressed: busy ? null : _installFromHub,
                icon: const Icon(Icons.download, size: 32),
                label: const Text(
                  'Pobierz model',
                  style: TextStyle(fontSize: 22),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Semantics(
            button: true,
            label: 'Wybierz plik modelu z pamięci telefonu',
            child: SizedBox(
              height: 72,
              child: OutlinedButton.icon(
                onPressed: busy ? null : _installFromFile,
                icon: const Icon(Icons.folder_open, size: 32),
                label: const Text(
                  'Wybierz plik .litertlm',
                  style: TextStyle(fontSize: 22),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Semantics(
            button: true,
            label: 'Wczytaj własny model SLAYER z folderu',
            child: SizedBox(
              height: 72,
              child: OutlinedButton.icon(
                onPressed: busy ? null : _loadSlayer,
                icon: const Icon(Icons.science, size: 32),
                label: const Text(
                  'Model SLAYER (folder z ONNX)',
                  style: TextStyle(fontSize: 22),
                ),
              ),
            ),
          ),
          const SizedBox(height: 24),
          if (_progress != null) ...[
            LinearProgressIndicator(value: _progress! > 0 ? _progress! / 100 : null),
            const SizedBox(height: 8),
            Text('Pobieranie: $_progress%', style: const TextStyle(fontSize: 18)),
          ],
          if (_error != null)
            Text(
              'Błąd: $_error',
              style: const TextStyle(fontSize: 18, color: Colors.redAccent),
            ),
        ],
      ),
    );
  }
}

