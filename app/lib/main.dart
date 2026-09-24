import 'package:flutter/material.dart';

import 'src/screens/home_screen.dart';
import 'src/screens/setup_screen.dart';
import 'src/services/gemma_service.dart';
import 'src/services/model_router.dart';
import 'src/services/slayer_service.dart';
import 'src/services/tts_service.dart';
import 'src/prompts.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await TtsService.instance.init();
  await GemmaService.instance.init();
  runApp(const VisionAssistantApp());
}

class VisionAssistantApp extends StatelessWidget {
  const VisionAssistantApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: StatusTexts.appTitle,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.indigo,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const StartupGate(),
    );
  }
}

/// Decyduje, czy model jest gotowy (Home), czy trzeba go pobrać (Setup).
class StartupGate extends StatefulWidget {
  const StartupGate({super.key});

  @override
  State<StartupGate> createState() => _StartupGateState();
}

class _StartupGateState extends State<StartupGate> {
  late final Future<void> _startup;

  @override
  void initState() {
    super.initState();
    _startup = _decide();
  }

  Future<void> _decide() async {
    // 1) Własny model SLAYER z katalogu aplikacji (jeśli pliki wgrane).
    if (await SlayerService.instance.tryAutoLoad()) {
      ModelRouter.engine = EngineKind.slayer;
      await TtsService.instance.speak(StatusTexts.modelReady);
      return;
    }
    // 2) Gemma 3n zainstalowana wcześniej.
    if (!GemmaService.instance.hasModel) {
      await TtsService.instance.speak(StatusTexts.modelMissing);
      return;
    }
    await GemmaService.instance.activate();
    await TtsService.instance.speak(StatusTexts.modelReady);
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _startup,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }
        if (GemmaService.instance.isReady || ModelRouter.slayerReady) {
          return const HomeScreen();
        }
        return const SetupScreen();
      },
    );
  }
}
