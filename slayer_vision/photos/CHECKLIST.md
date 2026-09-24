# Zbiór zdjęć — 200 scen (protokół)

Każda scena z `slayer_vision/scenes.py` czeka na zdjęcia w `photos/<scene_id>/`.
Builder sam tworzy brakujące katalogi — wystarczy raz uruchomić:

```bash
python -m slayer_vision.build_dataset --out out/dataset --per-kind 40 --photos photos
```

## Jak fotografować (kluczowe dla jakości modelu)

1. **3–5 zdjęć na scenę**, różne ujęcia (z góry, z boku, z bliska).
2. Odległość **20–50 cm** (przedmiot) / **30–60 cm** (dokument) — obiekt
   wypełnia kadr.
3. **Światło dzienne** przy oknie; bez lampy błyskowej (odbicia na folii).
4. Telefon trzymany stabilnie; obraz ostry (poczekaj na autofocus).
5. Format **JPG**, nazwy dowolne (builder sortuje i przepisuje).
6. Sceny z tekstem (dokumenty, etykiety): tekst prostopadle do obiektywu,
   bez cienia dłoni na treści.

Nie trzeba fotografować scen syntetycznych (paragon, data ważności, blister,
panel, pilot, list, metka) — te są renderowane z dokładnym GT. Zdjęcia
prawdziwych odpowiedników są jednak mile widziane jako weryfikacja domeny.

## Checklist

### Kuchnia i jedzenie (opakowania/ — 30)
- [ ] mleko · [ ] jogurt · [ ] ser · [ ] chleb · [ ] masło · [ ] sok
- [ ] woda · [ ] makaron · [ ] ryż · [ ] płatki · [ ] konserwa · [ ] dżem
- [ ] czekolada · [ ] kawa · [ ] herbata · [ ] przyprawy · [ ] mrożonka
- [ ] lody · [ ] baton · [ ] chipsy · [ ] keczup · [ ] majonez · [ ] olej
- [ ] ocet · [ ] mąka · [ ] cukier · [ ] sól · [ ] jajka · [ ] mięso · [ ] ryba

### Apteczka i zdrowie (leki/ — 20)
- [ ] blister · [ ] syrop · [ ] krople · [ ] maść · [ ] termometr
- [ ] ciśnieniomierz · [ ] plastry · [ ] witaminy · [ ] antybiotyk
- [ ] inhalator · [ ] strzykawka · [ ] maseczka · [ ] bandaże
- [ ] sól fizjologiczna · [ ] apteczka · [ ] okulary · [ ] bateria do aparatu
- [ ] pieluchy · [ ] chusteczki · [ ] termofor

### AGD i panele (agd/ — 25)
- [ ] pralka-panel · [ ] pralka-szuflada · [ ] pralka-bęben · [ ] zmywarka-panel
- [ ] zmywarka-koszyk · [ ] piekarnik-panel · [ ] mikrofalówka-panel
- [ ] lodówka-termostat · [ ] czajnik · [ ] ekspres · [ ] odkurzacz · [ ] żelazko
- [ ] wentylator · [ ] bojler · [ ] prysznic-bateria · [ ] zlew-kurek
- [ ] suszarka · [ ] robot kuchenny · [ ] blender · [ ] toster · [ ] waga
- [ ] termostat pokojowy · [ ] wideodomofon · [ ] domofon · [ ] winda-panel

### Piloty i sprzęt (piloty/ — 15)
- [ ] tv · [ ] dekoder · [ ] kino · [ ] klimatyzacja · [ ] projektor
- [ ] brama · [ ] światło · [ ] wentylator · [ ] klawiatura · [ ] mysz
- [ ] telefon klawiszowy · [ ] smartfon · [ ] tablet · [ ] budzik · [ ] zegar

### Dokumenty (dokumenty/ — 25) — własne/nieaktualne papiery, bez danych wrażliwych!
- [ ] list · [ ] awizo · [ ] wezwanie do zapłaty · [ ] rachunek-prąd
- [ ] rachunek-gaz · [ ] rachunek-woda · [ ] faktura · [ ] paragon · [ ] umowa
- [ ] wyciąg bankowy · [ ] PIT · [ ] decyzja ZUS · [ ] recepta · [ ] skierowanie
- [ ] wynik badania · [ ] legitymacja · [ ] dowód rejestracyjny · [ ] prawo jazdy
- [ ] dowód osobisty* · [ ] gwarancja · [ ] instrukcja · [ ] książeczka zdrowia
- [ ] zaproszenie · [ ] pocztówka · [ ] ulotka

\* dokumenty tożsamości — fotografuj **wyłącznie własne, nieaktualne** egzemplarze
albo pomiń; do treningu wystarczy sama kompozycja dokumentu.

### Zakupy i pieniądze (zakupy/ — 15)
- [ ] banknot · [ ] moneta · [ ] cena na półce · [ ] metka · [ ] kod kreskowy
- [ ] karta płatnicza · [ ] bilet komunikacji · [ ] bilet parkingowy
- [ ] karta lojalnościowa · [ ] portfel · [ ] lista zakupów · [ ] waga sklepowa
- [ ] kasa · [ ] koszyk · [ ] torba

### Ulica i transport (ulica/ — 20)
- [ ] przejście · [ ] sygnalizator · [ ] przystanek · [ ] rozkład · [ ] peron
- [ ] biletomat · [ ] nazwa ulicy · [ ] numer domu · [ ] drzwi wejściowe
- [ ] drzwi obrotowe · [ ] winda · [ ] schody · [ ] barierka · [ ] kosz
- [ ] hydrant · [ ] lampa · [ ] mapa · [ ] znak parking · [ ] pętla · [ ] tory

### Dom (dom/ — 20)
- [ ] drzwi-klamka · [ ] okno-klamka · [ ] szafa-uchwyt · [ ] zamek · [ ] klucz
- [ ] skrzynka pocztowa · [ ] gniazdko · [ ] wyłącznik · [ ] listwa · [ ] lampa
- [ ] półka · [ ] kosz na pranie · [ ] mop · [ ] wiadro · [ ] łóżko · [ ] stół
- [ ] krzesło · [ ] fotel · [ ] kanapa · [ ] szafka nocna

### Przesyłki (przesylki/ — 10)
- [ ] paczka · [ ] koperta · [ ] karton · [ ] taśma · [ ] etykieta adresowa
- [ ] naklejka „ostrożnie” · [ ] skrytka · [ ] paczkomat · [ ] torba prezentowa
- [ ] pudełko

### Różne (rozne/ — 20)
- [ ] książka · [ ] notes · [ ] długopis · [ ] okulary słoneczne · [ ] czapka
- [ ] rękawiczki · [ ] parasol · [ ] plecak · [ ] szalik · [ ] kluczyki
- [ ] ręcznik · [ ] szczoteczka · [ ] pasta · [ ] mydło · [ ] kubek · [ ] talerz
- [ ] sztućce · [ ] garnek · [ ] patelnia · [ ] lustro
