import '../prompts.dart';

// OCR podaje odczytane linie, a nie odpowiedź na pytanie. Kwotę i datę
// zwracamy wyłącznie, gdy rozpoznano odpowiednią etykietę na dokumencie.
String answerFromOcr(AskIntent intent, List<String> lines) {
  final nonEmpty = lines
      .map((line) => line.trim())
      .where((line) => line.isNotEmpty)
      .toList();
  if (intent == AskIntent.readText) {
    return nonEmpty.isEmpty
        ? 'Nie widzę czytelnego tekstu. Zbliż telefon i zrób ostre zdjęcie.'
        : 'Odczyt z aparatu, może zawierać błędy: ${nonEmpty.join('\n')}';
  }

  final folded = nonEmpty.map(_fold).toList();
  switch (intent) {
    case AskIntent.expiry:
      for (var i = 0; i < folded.length; i++) {
        final match = _expiryLabel.firstMatch(folded[i]);
        if (match == null) continue;
        final date =
            _dateOnLine(nonEmpty[i].substring(match.end)) ??
            (i + 1 < nonEmpty.length &&
                    !_otherDateLabel.hasMatch(folded[i + 1]) &&
                    !_expiryLabel.hasMatch(folded[i + 1])
                ? _dateOnLine(nonEmpty[i + 1])
                : null);
        if (date != null) return 'Odczyt OCR, sprawdź: data ważności $date.';
      }
      return 'Nie znalazłem czytelnej daty ważności. Zbliż aparat do oznaczenia EXP lub termin przydatności.';
    case AskIntent.payment:
      for (final label in _paymentLabels) {
        for (var i = 0; i < folded.length; i++) {
          final match = label.firstMatch(folded[i]);
          if (match == null ||
              _excludedPayment.hasMatch(folded[i].substring(0, match.start))) {
            continue;
          }
          final amount =
              _amountOnLine(nonEmpty[i].substring(match.end)) ??
              (i + 1 < nonEmpty.length &&
                      !_excludedPayment.hasMatch(folded[i + 1]) &&
                      !_paymentLabels.any(
                        (other) => other.hasMatch(folded[i + 1]),
                      )
                  ? _amountOnLine(nonEmpty[i + 1])
                  : null);
          if (amount != null) {
            return 'Odczyt OCR, sprawdź: do zapłaty $amount zł.';
          }
        }
      }
      return 'Nie znalazłem jednoznacznej kwoty do zapłaty. Zrób ostre zdjęcie całego paragonu.';
    case AskIntent.powerButton:
      for (final line in folded) {
        final match = _powerLabel.firstMatch(line);
        if (match != null) {
          return 'Widzę napis ${match.group(1)}. Poszukaj przycisku z tym napisem. Nie potrafię potwierdzić jego położenia ani rozpoznać samej ikony zasilania.';
        }
      }
      return 'Nie widzę czytelnego napisu zasilania. Nie potrafię wskazać przycisku na podstawie samej ikony; zbliż aparat do panelu.';
    case AskIntent.scene:
    case AskIntent.object:
      throw ArgumentError('Opis sceny nie jest zadaniem OCR');
    case AskIntent.readText:
      throw StateError('Obsłużone przed OCR');
  }
}

String _fold(String text) {
  const from = 'ĄĆĘŁŃÓŚŹŻąćęłńóśźż';
  const to = 'ACELNOSZZacelnoszz';
  final buffer = StringBuffer();
  for (final rune in text.runes) {
    final ch = String.fromCharCode(rune);
    final index = from.indexOf(ch);
    buffer.write(index < 0 ? ch.toUpperCase() : to[index].toUpperCase());
  }
  return buffer.toString();
}

final _expiryLabel = RegExp(
  r'\b(?:EXP(?:IRY)?|DATA WAZNOSCI|TERMIN PRZYDATNOSCI|WAZNE DO|NAJLEPIEJ SPOZYC PRZED|SPOZYC DO|BEST BEFORE|USE BY)\b',
);
final _otherDateLabel = RegExp(
  r'\b(?:PRODUKCJ\w*|WYPRODUKOWANO|MFG|LOT|BATCH|VAT)\b',
);
final _paymentLabels = [
  RegExp(r'\bDO ZAPLATY\b'),
  RegExp(r'\bNALEZNOSC\b'),
  RegExp(r'\bSUMA\b'),
  RegExp(r'\bRAZEM\b'),
  RegExp(r'\bTOTAL\b'),
];
final _excludedPayment = RegExp(
  r'\b(?:VAT|PODATEK|RESZTA|WYDAC|GOTOWKA|RABAT|PRZEDPLATA)\b',
);
final _powerLabel = RegExp(r'\b(ON[ /-]*OFF|POWER|ZASILANIE|WLACZ|WYLACZ)\b');

String? _dateOnLine(String line) {
  final yearFirst = RegExp(
    r'(?<!\d)(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)',
  ).firstMatch(line);
  if (yearFirst != null) {
    final year = int.parse(yearFirst.group(1)!);
    final month = int.parse(yearFirst.group(2)!);
    final day = int.parse(yearFirst.group(3)!);
    if (_validDate(year, month, day)) return yearFirst.group(0);
    return null;
  }
  final dayFirst = RegExp(
    r'(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2}|20\d{2})(?!\d)',
  ).firstMatch(line);
  if (dayFirst != null) {
    final day = int.parse(dayFirst.group(1)!);
    final month = int.parse(dayFirst.group(2)!);
    final yearValue = int.parse(dayFirst.group(3)!);
    final year = yearValue < 100 ? yearValue + 2000 : yearValue;
    if (_validDate(year, month, day)) return dayFirst.group(0);
    return null;
  }
  final monthYear = RegExp(r'(?<![\d./-])(\d{1,2})[./-](20\d{2})(?!\d)')
      .firstMatch(line);
  if (monthYear != null &&
      int.parse(monthYear.group(1)!) <= 12 &&
      int.parse(monthYear.group(1)!) > 0) {
    return monthYear.group(0);
  }
  // Na opakowaniach EXP 05/27 oznacza miesiąc i rok, nie piąty dzień.
  final shortYear = RegExp(r'(?<![\d./-])(\d{1,2})[./-](\d{2})(?!\d)')
      .firstMatch(line);
  if (shortYear != null &&
      int.parse(shortYear.group(1)!) >= 1 &&
      int.parse(shortYear.group(1)!) <= 12) {
    return shortYear.group(0);
  }
  return null;
}

bool _validDate(int year, int month, int day) {
  if (month < 1 || month > 12 || day < 1 || day > 31) return false;
  final date = DateTime(year, month, day);
  return date.year == year && date.month == month && date.day == day;
}

String? _amountOnLine(String line) {
  final match = RegExp(
    r'(?<!\d)(\d{1,3}(?:[ \u00a0]\d{3})*|\d+)[,.](\d{2})(?!\d)',
  ).firstMatch(line);
  if (match == null) return null;
  final whole = match.group(1)!.replaceAll(RegExp(r'[ \u00a0]'), '');
  return '$whole,${match.group(2)}';
}
