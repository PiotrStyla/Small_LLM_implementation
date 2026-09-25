import 'package:flutter_test/flutter_test.dart';
import 'package:vision_assistant/src/prompts.dart';
import 'package:vision_assistant/src/services/ocr_answers.dart';

void main() {
  test('reads all recognized text rather than an image caption', () {
    expect(
      answerFromOcr(AskIntent.readText, ['  Mleko  ', '2 litry']),
      contains('Mleko\n2 litry'),
    );
  });

  test('restores Polish diacritics lost by OCR in known words', () {
    expect(
      answerFromOcr(AskIntent.readText, ['Waznosc 05.07.2027']),
      contains('Ważność 05.07.2027'),
    );
    expect(
      answerFromOcr(AskIntent.readText, ['SKLAD: woda, cukier, sol']),
      contains('SKŁAD: woda, cukier, sól'),
    );
    // Nieznane wyrazy zostają dokładnie tak, jak je odczytano.
    expect(
      answerFromOcr(AskIntent.readText, ['Xqwert 123']),
      contains('Xqwert 123'),
    );
  });

  test('expiry requires the label and a valid associated date', () {
    expect(
      answerFromOcr(AskIntent.expiry, [
        'Data produkcji 01.03.2026',
        'EXP: 28.02.2027',
      ]),
      allOf(contains('28.02.2027'), isNot(contains('01.03.2026'))),
    );
    expect(
      answerFromOcr(AskIntent.expiry, ['EXP:', '05/27']),
      contains('05/27'),
    );
    expect(
      answerFromOcr(AskIntent.expiry, ['EXP: 31.02.2027']),
      isNot(contains('31.02.2027')),
    );
    expect(
      answerFromOcr(AskIntent.expiry, ['Data produkcji 01.03.2026']),
      isNot(contains('01.03.2026')),
    );
    expect(
      answerFromOcr(AskIntent.expiry, [
        'Data produkcji 01.03.2026 EXP 20.05.2027',
      ]),
      allOf(contains('20.05.2027'), isNot(contains('01.03.2026'))),
    );
  });

  test('payment prefers amount due over VAT or an earlier subtotal', () {
    expect(
      answerFromOcr(AskIntent.payment, [
        'SUMA VAT 4,60',
        'SUMA 19,90',
        'DO ZAPŁATY',
        '23,00 zł',
        'RESZTA 7,00',
      ]),
      allOf(
        contains('23,00'),
        isNot(contains('19,90')),
        isNot(contains('4,60')),
      ),
    );
    expect(
      answerFromOcr(AskIntent.payment, ['VAT 4,60', 'Reszta 7,00']),
      allOf(isNot(contains('4,60')), isNot(contains('7,00'))),
    );
    expect(
      answerFromOcr(AskIntent.payment, ['RAZEM 1 234,56']),
      contains('1234,56'),
    );
  });

  test(
    'power button requires a visible label and never invents a location',
    () {
      expect(
        answerFromOcr(AskIntent.powerButton, ['POWER', 'MODE']),
        contains('napis POWER'),
      );
      expect(
        answerFromOcr(AskIntent.powerButton, ['MODE', 'OK']),
        isNot(contains('POWER')),
      );
      // Oryginalna pisownia z OCR (z polskimi znakami), nie wariant złożony.
      expect(
        answerFromOcr(AskIntent.powerButton, ['WŁĄCZ WYŁĄCZ']),
        allOf(contains('napis WŁĄCZ'), isNot(contains('WLACZ'))),
      );
    },
  );

  test('voice maps only supported tasks', () {
    expect(intentForVoice('Przeczytaj tekst'), AskIntent.readText);
    expect(intentForVoice('Ile do zapłaty?'), AskIntent.payment);
    expect(intentForVoice('Gdzie są klucze?'), isNull);
  });
}
