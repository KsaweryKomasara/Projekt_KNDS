# Projekt_KNDS
Projekt realizowany w ramach koła naukowego Data Science
Autorzy: Ksawery Komasara, Mateusz Marcinkowski, Jakub Gronkiewicz

Kluczowe wnioski, najważniejsze decyzje ze wszystkich etapów, wykresy, tabele porównujące jakość modeli.

1. Eksproracyjna analiza danych

Zmienna celu: booking_status
Zmienna celu przyjmuje dwie wartości: Canceled / Not_canceled

Typy danych: object, int64, float64

Brak duplikatów

Brak wartości pustych

Do zastanowienia: Wg IRQAnalises występuje sporo wartości odstających

Do zastanowienia: Wykres korelacji zmiennych ze zmienną celu (Ważny wykres)

Wykres ze zobrazowaniem liczby wystąpień rezerwacji dla konkretnych cen pokojów hotelowych pokazał, że istnieje ponad 500 rezerwacji ze średnią ceną pokoju równą 0. Wynika to z dwóch czynników. Cena za pokój jest równa zero kiedy typ gościa hotelowego (zmienna market_segment_type) przyjumuje wartość "Complimetary", co oznacza, że rezerwacja nie jest typową rezerwacją i cena za pokój wynosi 0. Cena za pokój wyniesie 0 również gdy klient nie podał ani liczby nocy w tygodniu roboczym, ani liczby nocy weekendowych.

2. Przetwarzanie danych

Mapowanie kategorycznej zmiennej celu: Canceled - 1, Not_canceled - 0

Po wykonaniu preprocessingu w danych było aż 30 tysięcy kolumn. Macierz zredukowano do 30 kolumn po dropnięciu kolumny Booking_ID.

3. Trening i ewaluacja modeli

Wykorzystano 4 modele: regresję logistyczną, drzewo decyzyjne, random forest oraz adaBoost

4. Intrepretacja modelu

Regresja logistyczna

Drzewo decyzyjne

Random forest

AdaBoost

5. Podsumowanie

Tu wskazanie na tabelkę, która porównuje wszystkie wykorzystane modele.
