"""Zamknięty katalog scen produktu „pokaż i zapytaj" (wersja PL).

Katalog definiuje, co model MUSI rozpoznawać — i jedyne, co ma mówić.
Zamknięcie katalogu jest świadomym kompromisem: 110M model halucynuje
w otwartym „opisz świat", a użytkownik potrzebuje przewidywalnych zdań.

Dwa użycia:
- sceny fotograficzne: użytkownik robi zdjęcia `photos/<id>/*.jpg`,
  zdanie docelowe bierzemy z katalogu (patrz `build_dataset.py`);
- sceny syntetyczne (dokumenty/panele): obrazy RENDERUJEMY z pełnym GT,
  zdania ze slotami generuje `slayer_vision.synth` (np. kwota z rachunku).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    id: str  # katalog/obiekt — używane jako katalog zdjęć
    category: str
    name: str
    sentence: str  # zdanie docelowe dla scen fotograficznych


def _rows(category: str, rows: list[tuple[str, str, str]]) -> list[Scene]:
    return [Scene(rid, category, name, sentence) for rid, name, sentence in rows]


# ---------------------------------------------------------------------------
# Sceny fotograficzne — użytkownik/zbieracz robi prawdziwe zdjęcia.
# ---------------------------------------------------------------------------

FOOD = _rows("Opakowania żywności", [
    ("opakowania/mleko", "karton mleka", "To karton mleka."),
    ("opakowania/jogurt", "kubeczek jogurtu", "To kubeczek jogurtu."),
    ("opakowania/ser", "opakowanie sera żółtego", "To opakowanie sera żółtego."),
    ("opakowania/chleb", "bochenek chleba w worku", "To bochenek chleba w worku."),
    ("opakowania/maslo", "kostka masła", "To kostka masła."),
    ("opakowania/sok", "karton soku", "To karton soku."),
    ("opakowania/woda", "butelka wody mineralnej", "To butelka wody mineralnej."),
    ("opakowania/makaron", "opakowanie makaronu", "To opakowanie makaronu."),
    ("opakowania/ryz", "opakowanie ryżu", "To opakowanie ryżu."),
    ("opakowania/platki", "pudełko płatków śniadaniowych", "To pudełko płatków śniadaniowych."),
    ("opakowania/konserwa", "puszka konserwy", "To puszka konserwy."),
    ("opakowania/dzem", "słoik dżemu", "To słoik dżemu."),
    ("opakowania/czekolada", "tabliczka czekolady", "To tabliczka czekolady."),
    ("opakowania/kawa", "opakowanie kawy", "To opakowanie kawy."),
    ("opakowania/herbata", "pudełko herbaty", "To pudełko herbaty."),
    ("opakowania/przyprawy", "słoiczek z przyprawą", "To słoiczek z przyprawą."),
    ("opakowania/mrozonka", "opakowanie mrożonki", "To opakowanie mrożonki."),
    ("opakowania/lody", "opakowanie lodów", "To opakowanie lodów."),
    ("opakowania/baton", "batonik", "To batonik."),
    ("opakowania/chipsy", "opakowanie chipsów", "To opakowanie chipsów."),
    ("opakowania/keczup", "butelka keczupu", "To butelka keczupu."),
    ("opakowania/majonez", "słoik majonezu", "To słoik majonezu."),
    ("opakowania/olej", "butelka oleju", "To butelka oleju."),
    ("opakowania/ocet", "butelka octu", "To butelka octu."),
    ("opakowania/maka", "opakowanie mąki", "To opakowanie mąki."),
    ("opakowania/cukier", "opakowanie cukru", "To opakowanie cukru."),
    ("opakowania/sol", "opakowanie soli", "To opakowanie soli."),
    ("opakowania/jajka", "wytłaczanka na jajka", "To wytłaczanka na jajka."),
    ("opakowania/mieso", "opakowane mięso", "To opakowane mięso."),
    ("opakowania/ryba", "opakowana ryba", "To opakowana ryba."),
])

HEALTH = _rows("Leki i zdrowie", [
    ("leki/blister", "blister tabletek", "To blister tabletek."),
    ("leki/syrop", "butelka syropu", "To butelka syropu."),
    ("leki/krople", "buteleczka kropli", "To buteleczka kropli."),
    ("leki/masc", "tubka maści", "To tubka maści."),
    ("leki/termometr", "termometr", "To termometr."),
    ("leki/cisnieniomierz", "ciśnieniomierz", "To ciśnieniomierz."),
    ("leki/plastry", "opakowanie plastrów", "To opakowanie plastrów."),
    ("leki/witaminy", "opakowanie witamin", "To opakowanie witamin."),
    ("leki/antybiotyk", "opakowanie antybiotyku", "To opakowanie antybiotyku."),
    ("leki/inhalator", "inhalator", "To inhalator."),
    ("leki/strzykawka", "strzykawka", "To strzykawka."),
    ("leki/maseczka", "maseczka ochronna", "To maseczka ochronna."),
    ("leki/bandaze", "opakowanie bandaży", "To opakowanie bandaży."),
    ("leki/sol-fizjologiczna", "ampułka soli fizjologicznej", "To ampułka soli fizjologicznej."),
    ("leki/apteczka", "apteczka pierwszej pomocy", "To apteczka pierwszej pomocy."),
    ("leki/okulary", "okulary w etui", "To okulary w etui."),
    ("leki/bateria-aparat-sluchowy", "bateria do aparatu słuchowego", "To bateria do aparatu słuchowego."),
    ("leki/pampersy", "opakowanie pieluch dla dorosłych", "To opakowanie pieluch dla dorosłych."),
    ("leki/chusteczki", "pudełko chusteczek", "To pudełko chusteczek."),
    ("leki/termofor", "termofor", "To termofor."),
])

APPLIANCE = _rows("AGD i panele", [
    ("agd/pralka-panel", "panel pralki", "To panel pralki."),
    ("agd/pralka-szuflada", "szuflada na proszek w pralce", "To szuflada na proszek w pralce."),
    ("agd/pralka-buben", "bęben pralki", "To otwarty bęben pralki."),
    ("agd/zmywarka-panel", "panel zmywarki", "To panel zmywarki."),
    ("agd/zmywarka-koszyk", "koszyk zmywarki", "To wysunięty koszyk zmywarki."),
    ("agd/piekarnik-panel", "panel piekarnika", "To panel piekarnika."),
    ("agd/mikrofalowka-panel", "panel mikrofalówki", "To panel mikrofalówki."),
    ("agd/lodowka-termostat", "pokrętło termostatu w lodówce", "To pokrętło termostatu w lodówce."),
    ("agd/czajnik", "czajnik elektryczny", "To czajnik elektryczny."),
    ("agd/ekspres", "ekspres do kawy", "To ekspres do kawy."),
    ("agd/odkurzacz", "odkurzacz", "To odkurzacz."),
    ("agd/zelazko", "żelazko", "To żelazko."),
    ("agd/wentylator", "wentylator", "To wentylator."),
    ("agd/bojler", "bojler z pokrętłem", "To bojler z pokrętłem."),
    ("agd/prysznic-bateria", "bateria prysznicowa", "To bateria prysznicowa."),
    ("agd/zlew-kurek", "kurek przy zlewie", "To kurek przy zlewie."),
    ("agd/suszarka", "suszarka do włosów", "To suszarka do włosów."),
    ("agd/robot-kuchenny", "robot kuchenny", "To robot kuchenny."),
    ("agd/blender", "blender", "To blender."),
    ("agd/toster", "toster", "To toster."),
    ("agd/waga", "waga łazienkowa", "To waga łazienkowa."),
    ("agd/termostat-pokojowy", "termostat pokojowy", "To termostat pokojowy."),
    ("agd/wideodomofon", "wideodomofon", "To wideodomofon."),
    ("agd/domofon", "domofon", "To domofon."),
    ("agd/winda-panel", "panel windy", "To panel windy."),
])

REMOTE = _rows("Piloty i klawiatury", [
    ("piloty/tv", "pilot od telewizora", "To pilot od telewizora."),
    ("piloty/dekoder", "pilot od dekodera", "To pilot od dekodera."),
    ("piloty/kino", "pilot od kina domowego", "To pilot od kina domowego."),
    ("piloty/klimatyzacja", "pilot od klimatyzacji", "To pilot od klimatyzacji."),
    ("piloty/projektor", "pilot od projektora", "To pilot od projektora."),
    ("piloty/brama", "pilot do bramy wjazdowej", "To pilot do bramy wjazdowej."),
    ("piloty/swiatlo", "pilot do światła", "To pilot do światła."),
    ("piloty/wentylator", "pilot od wentylatora", "To pilot od wentylatora."),
    ("piloty/klawiatura", "klawiatura komputerowa", "To klawiatura komputerowa."),
    ("piloty/mysz", "mysz komputerowa", "To mysz komputerowa."),
    ("piloty/telefon-klawiszowy", "telefon z klawiszami", "To telefon z klawiszami."),
    ("piloty/smartfon", "smartfon", "To smartfon."),
    ("piloty/tablet", "tablet", "To tablet."),
    ("piloty/budzik", "budzik", "To budzik."),
    ("piloty/zegar", "zegar ścienny", "To zegar ścienny."),
])

# Sceny dokumentów — DOMYŚLNIE syntetyczne (patrz `synth.SYNTH_KINDS`),
# ale katalog dopuszcza też zdjęcia prawdziwych dokumentów.
DOCS = _rows("Dokumenty i korespondencja", [
    ("dokumenty/list", "list w kopercie", "To list w kopercie."),
    ("dokumenty/list-polecony", "awizo listu poleconego", "To awizo listu poleconego."),
    ("dokumenty/wezwanie", "wezwanie do zapłaty", "To wezwanie do zapłaty."),
    ("dokumenty/rachunek-prad", "rachunek za prąd", "To rachunek za prąd."),
    ("dokumenty/rachunek-gaz", "rachunek za gaz", "To rachunek za gaz."),
    ("dokumenty/rachunek-woda", "rachunek za wodę", "To rachunek za wodę."),
    ("dokumenty/faktura", "faktura", "To faktura."),
    ("dokumenty/paragon", "paragon ze sklepu", "To paragon ze sklepu."),
    ("dokumenty/umowa", "umowa", "To umowa."),
    ("dokumenty/wyciag-bankowy", "wyciąg z banku", "To wyciąg z banku."),
    ("dokumenty/pit", "deklaracja podatkowa PIT", "To deklaracja podatkowa PIT."),
    ("dokumenty/zus", "decyzja z ZUS", "To decyzja z ZUS."),
    ("dokumenty/recepta", "recepta", "To recepta."),
    ("dokumenty/skierowanie", "skierowanie na badanie", "To skierowanie na badanie."),
    ("dokumenty/wynik-badania", "wynik badania", "To wynik badania."),
    ("dokumenty/legitymacja", "legitymacja", "To legitymacja."),
    ("dokumenty/dowod-rejestracyjny", "dowód rejestracyjny", "To dowód rejestracyjny."),
    ("dokumenty/prawo-jazdy", "prawo jazdy", "To prawo jazdy."),
    ("dokumenty/dowod-osobisty", "dowód osobisty", "To dowód osobisty."),
    ("dokumenty/gwarancja", "karta gwarancyjna", "To karta gwarancyjna."),
    ("dokumenty/instrukcja", "instrukcja obsługi", "To instrukcja obsługi."),
    ("dokumenty/ksiazeczka-zdrowia", "książeczka zdrowia", "To książeczka zdrowia."),
    ("dokumenty/zaproszenie", "zaproszenie", "To zaproszenie."),
    ("dokumenty/pocztowka", "pocztówka", "To pocztówka."),
    ("dokumenty/ulotka", "ulotka", "To ulotka."),
])

MONEY = _rows("Pieniądze i zakupy", [
    ("zakupy/banknot", "banknot", "To banknot."),
    ("zakupy/moneta", "moneta", "To moneta."),
    ("zakupy/cena-polka", "etykieta z ceną na półce", "To etykieta z ceną na półce."),
    ("zakupy/metka", "metka na odzieży", "To metka na odzieży."),
    ("zakupy/kod-kreskowy", "kod kreskowy", "To kod kreskowy."),
    ("zakupy/karta-platnicza", "karta płatnicza", "To karta płatnicza."),
    ("zakupy/bilet-komunikacji", "bilet komunikacji miejskiej", "To bilet komunikacji miejskiej."),
    ("zakupy/bilet-parkingowy", "bilet parkingowy", "To bilet parkingowy."),
    ("zakupy/karta-lojalnosciowa", "karta lojalnościowa", "To karta lojalnościowa."),
    ("zakupy/portfel", "portfel", "To portfel."),
    ("zakupy/lista-zakupow", "lista zakupów", "To lista zakupów."),
    ("zakupy/waga-sklepowa", "waga sklepowa", "To waga sklepowa."),
    ("zakupy/kasa", "kasa samoobsługowa", "To kasa samoobsługowa."),
    ("zakupy/koszyk", "koszyk sklepowy", "To koszyk sklepowy."),
    ("zakupy/torba", "torba z zakupami", "To torba z zakupami."),
])

STREET = _rows("Ulica i transport", [
    ("ulica/przejscie", "przejście dla pieszych", "Przejście dla pieszych znajduje się przed tobą."),
    ("ulica/sygnalizator", "sygnalizator świetlny", "To sygnalizator świetlny."),
    ("ulica/przystanek", "przystanek autobusowy", "To przystanek autobusowy."),
    ("ulica/rozklad", "rozkład jazdy", "To rozkład jazdy."),
    ("ulica/peron", "tablica z numerem peronu", "To tablica z numerem peronu."),
    ("ulica/biletomat", "biletomat", "To biletomat."),
    ("ulica/nazwa-ulicy", "tabliczka z nazwą ulicy", "To tabliczka z nazwą ulicy."),
    ("ulica/numer-domu", "numer na bramie", "To numer na bramie."),
    ("ulica/drzwi-wejsciowe", "drzwi wejściowe", "To drzwi wejściowe."),
    ("ulica/drzwi-obrotowe", "drzwi obrotowe", "To drzwi obrotowe."),
    ("ulica/winda", "winda", "To winda."),
    ("ulica/schody", "schody z poręczą", "Schody prowadzą w górę, poręcz jest po prawej stronie."),
    ("ulica/barierka", "barierka", "To barierka."),
    ("ulica/kosz", "kosz na śmieci", "To kosz na śmieci."),
    ("ulica/hydrant", "hydrant", "To hydrant."),
    ("ulica/lampa", "latarnia uliczna", "To latarnia uliczna."),
    ("ulica/mapa", "mapa miasta", "To mapa miasta."),
    ("ulica/znak-parking", "znak parkingowy", "To znak parkingowy."),
    ("ulica/loopka", "pętla autobusowa", "To pętla autobusowa."),
    ("ulica/przejscie-kolejowe", "przejście przez tory", "To przejście przez tory."),
])

HOME = _rows("Dom codzienny", [
    ("dom/drzwi-klamka", "klamka od drzwi", "Klamka jest po lewej stronie drzwi."),
    ("dom/okno-klamka", "klamka od okna", "Klamka okna jest po prawej stronie."),
    ("dom/szafa-uchwyt", "uchwyt szafy", "To uchwyt szafy."),
    ("dom/zamek", "zamek w drzwiach", "To zamek w drzwiach."),
    ("dom/klucz", "klucz", "To klucz."),
    ("dom/skrzynka-pocztowa", "skrzynka pocztowa", "To skrzynka pocztowa."),
    ("dom/gniazdko", "gniazdko elektryczne", "To gniazdko elektryczne."),
    ("dom/wylacznik", "wyłącznik światła", "To wyłącznik światła."),
    ("dom/listwa", "listwa zasilająca", "To listwa zasilająca."),
    ("dom/lampa-stojaca", "lampa stojąca", "To lampa stojąca."),
    ("dom/polka", "półka", "To półka."),
    ("dom/kosz-pranie", "kosz na pranie", "To kosz na pranie."),
    ("dom/mop", "mop", "To mop."),
    ("dom/wiadro", "wiadro", "To wiadro."),
    ("dom/lozko", "łóżko", "To łóżko."),
    ("dom/stol", "stół", "To stół."),
    ("dom/krzeslo", "krzesło", "To krzesło."),
    ("dom/fotel", "fotel", "To fotel."),
    ("dom/kanapa", "kanapa", "To kanapa."),
    ("dom/szafka-nocna", "szafka nocna", "To szafka nocna."),
])

SHIPMENTS = _rows("Przesyłki i opakowania", [
    ("przesylki/paczka", "paczka kurierska", "To paczka kurierska."),
    ("przesylki/koperta", "koperta z listem", "To koperta z listem."),
    ("przesylki/karton", "karton zbiorczy", "To karton zbiorczy."),
    ("przesylki/tasma", "taśma klejąca", "To taśma klejąca."),
    ("przesylki/etykieta-adresowa", "etykieta adresowa", "To etykieta adresowa."),
    ("przesylki/ostroznosc", "naklejka „ostrożnie”", "To naklejka „ostrożnie”."),
    ("przesylki/skrytka", "skrytka pocztowa", "To skrytka pocztowa."),
    ("przesylki/paczkomat", "paczkomat", "To paczkomat."),
    ("przesylki/torba-prezentowa", "torba prezentowa", "To torba prezentowa."),
    ("przesylki/pudelko", "puste pudełko", "To puste pudełko."),
])

MISC = _rows("Różne", [
    ("rozne/ksiazka", "książka", "To książka."),
    ("rozne/notes", "notes", "To notes."),
    ("rozne/dlugopis", "długopis", "To długopis."),
    ("rozne/okulary-sloneczne", "okulary przeciwsłoneczne", "To okulary przeciwsłoneczne."),
    ("rozne/czapka", "czapka", "To czapka."),
    ("rozne/rekawiczki", "rękawiczki", "To rękawiczki."),
    ("rozne/parasol", "parasol", "To parasol."),
    ("rozne/plecak", "plecak", "To plecak."),
    ("rozne/szalik", "szalik", "To szalik."),
    ("rozne/kluczyki-samochodowe", "kluczyki do samochodu", "To kluczyki do samochodu."),
    ("rozne/recznik", "ręcznik", "To ręcznik."),
    ("rozne/szczoteczka", "szczoteczka do zębów", "To szczoteczka do zębów."),
    ("rozne/pasta", "tubka pasty do zębów", "To tubka pasty do zębów."),
    ("rozne/mydlo", "mydelniczka z mydłem", "To mydelniczka z mydłem."),
    ("rozne/kubek", "kubek", "To kubek."),
    ("rozne/talerz", "talerz", "To talerz."),
    ("rozne/sztucce", "sztućce", "To sztućce."),
    ("rozne/garnek", "garnek", "To garnek."),
    ("rozne/patelnia", "patelnia", "To patelnia."),
    ("rozne/lustro", "lustro", "To lustro."),
    ("rozne/sluchawki", "słuchawki", "To słuchawki."),
])

INTERIOR = _rows("Wnętrza", [
    ("wnetrza/korytarz", "korytarz", "To korytarz."),
    ("wnetrza/przedpokoj", "przedpokój", "To przedpokój."),
    ("wnetrza/pokoj", "pokój", "To pokój."),
    ("wnetrza/salon", "salon", "To salon."),
    ("wnetrza/kuchnia", "kuchnia", "To kuchnia."),
    ("wnetrza/lazienka", "łazienka", "To łazienka."),
    ("wnetrza/schody", "schody", "To schody."),
])

PLANTS = _rows("Rośliny i kwiaty", [
    ("rosliny/kwiaty-doniczkowe", "kwiaty w doniczce", "To kwiaty w doniczce."),
    ("rosliny/roslina-doniczkowa", "roślina doniczkowa", "To roślina doniczkowa."),
    ("rosliny/kwiaty-wazonie", "kwiaty w wazonie", "To kwiaty w wazonie."),
    ("rosliny/kwiaty-na-mebelu", "kwiaty na szafce", "To kwiaty na szafce."),
])

SCENES: tuple[Scene, ...] = tuple(FOOD + HEALTH + APPLIANCE + REMOTE + DOCS + MONEY + STREET + HOME + SHIPMENTS + MISC + INTERIOR + PLANTS)

CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(scene.category for scene in SCENES))

assert len(SCENES) == 212, f"Katalog ma {len(SCENES)} scen, oczekiwano 212"
assert len({scene.id for scene in SCENES}) == len(SCENES), "Powtórzone id sceny"
