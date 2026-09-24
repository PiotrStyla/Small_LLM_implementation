import 'package:flutter/material.dart';

/// Predefiniowane pytania dla trybu „pokaż i zapytaj".
///
/// Zakres jest celowo zamknięty: model ma odpowiadać krótko i przewidywalnie,
/// a użytkownik nie musi uczyć się formułowania promptów.
class AskPreset {
  const AskPreset(this.label, this.question, this.icon);

  /// Etykieta na przycisku (mówiona przez TTS i czytana przez TalkBack).
  final String label;

  /// Treść zapytania wysyłanego do modelu wraz ze zdjęciem.
  final String question;

  final IconData icon;
}

/// Stały suffix stylu — krótkie, jednozdaniowe odpowiedzi po polsku.
const String kStyleSuffix =
    ' Odpowiedz po polsku, jednym krótkim zdaniem. Bez list, bez formatowania, bez dodatkowych komentarzy.';

const List<AskPreset> kPresets = [
  AskPreset(
    'Co jest przede mną?',
    'Opisz co widzisz na tym zdjęciu.',
    Icons.center_focus_strong,
  ),
  AskPreset(
    'Co to jest?',
    'Co to za przedmiot?',
    Icons.help_outline,
  ),
  AskPreset(
    'Przeczytaj tekst',
    'Przeczytaj cały tekst widoczny na tym zdjęciu.',
    Icons.menu_book,
  ),
  AskPreset(
    'Data ważności',
    'Jaka jest data ważności lub termin przydatności na tym opakowaniu?',
    Icons.event,
  ),
  AskPreset(
    'Ile do zapłaty?',
    'Ile mam zapłacić według tego rachunku? Podaj kwotę.',
    Icons.payments,
  ),
  AskPreset(
    'Który przycisk?',
    'Który przycisk na tym urządzeniu służy do włączania? Opisz jego położenie.',
    Icons.smart_button,
  ),
];

/// Buduje pełne zapytanie: pytanie użytkownika + wymuszenie stylu odpowiedzi.
String buildQuestion(String question) => '$question$kStyleSuffix';

/// Komunikaty statusu — czytane na ekranie i wypowiadane przez TTS.
abstract final class StatusTexts {
  static const String appTitle = 'Asystent wzrokowy';
  static const String askButton = 'Pokaż i zapytaj';
  static const String askButtonHint =
      'Robi zdjęcie i opisuje co znajduje się przed kamerą';
  static const String micButton = 'Zadaj pytanie głosem';
  static const String listening = 'Słucham pytania…';
  static const String capturing = 'Robię zdjęcie…';
  static const String thinking = 'Analizuję zdjęcie…';
  static const String modelMissing = 'Model nie jest jeszcze pobrany.';
  static const String modelReady = 'Model gotowy. Możesz robić zdjęcia.';
  static const String noCamera = 'Brak dostępu do kamery.';
  static const String noAnswer = 'Nie udało się uzyskać odpowiedzi.';
  static const String noSpeech = 'Nie rozpoznałem pytania.';
}
