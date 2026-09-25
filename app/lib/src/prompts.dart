import 'package:flutter/material.dart';

/// SLAYER jest modelem podpisów obrazów; zadania tekstowe obsługuje OCR.
enum AskIntent { scene, object, readText, expiry, payment, powerButton }

class AskPreset {
  const AskPreset(this.label, this.question, this.icon, this.intent);

  final String label;
  final String question;
  final IconData icon;
  final AskIntent intent;
}

/// Stały suffix stylu — krótkie, jednozdaniowe odpowiedzi po polsku.
const String kStyleSuffix =
    ' Odpowiedz po polsku, jednym krótkim zdaniem. Bez list, bez formatowania, bez dodatkowych komentarzy.';

const List<AskPreset> kPresets = [
  AskPreset(
    'Co jest przede mną?',
    'Opisz co widzisz na tym zdjęciu.',
    Icons.center_focus_strong,
    AskIntent.scene,
  ),
  AskPreset(
    'Co to jest?',
    'Co to za przedmiot?',
    Icons.help_outline,
    AskIntent.object,
  ),
  AskPreset(
    'Przeczytaj tekst',
    'Przeczytaj cały tekst widoczny na tym zdjęciu.',
    Icons.menu_book,
    AskIntent.readText,
  ),
  AskPreset(
    'Data ważności',
    'Jaka jest data ważności lub termin przydatności na tym opakowaniu?',
    Icons.event,
    AskIntent.expiry,
  ),
  AskPreset(
    'Ile do zapłaty?',
    'Ile mam zapłacić według tego rachunku? Podaj kwotę.',
    Icons.payments,
    AskIntent.payment,
  ),
  AskPreset(
    'Który przycisk?',
    'Który przycisk na tym urządzeniu służy do włączania? Opisz jego położenie.',
    Icons.smart_button,
    AskIntent.powerButton,
  ),
];

/// Pytania głosowe SLAYER: rozpoznaj wyłącznie znane polecenia.
/// Innych pytań model podpisów obrazów nie potrafi wykonać.
AskIntent? intentForVoice(String question) {
  final text = question.toLowerCase();
  if (text.contains('ważnoś') || text.contains('termin przydatnoś')) {
    return AskIntent.expiry;
  }
  if (text.contains('zapłaci') ||
      text.contains('do zapłaty') ||
      text.contains('paragon') ||
      text.contains('rachunek')) {
    return AskIntent.payment;
  }
  if (text.contains('przycisk') || text.contains('włącz')) {
    return AskIntent.powerButton;
  }
  if (text.contains('przeczytaj') ||
      text.contains('odczytaj') ||
      text.contains('tekst') ||
      text.contains('napis')) {
    return AskIntent.readText;
  }
  if (text.contains('co to jest') || text.contains('jaki to przedmiot')) {
    return AskIntent.object;
  }
  if (text.contains('przede mną') ||
      text.contains('co widzisz') ||
      text.contains('opisz')) {
    return AskIntent.scene;
  }
  return null;
}

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
  static const String modelReady =
      'Model gotowy. Small LLM by Fabryka AI. Możesz robić zdjęcia.';
  static const String noCamera = 'Brak dostępu do kamery.';
  static const String noAnswer = 'Nie udało się uzyskać odpowiedzi.';
  static const String noSpeech = 'Nie rozpoznałem pytania.';
}
