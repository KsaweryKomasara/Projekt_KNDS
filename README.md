# Projekt_KNDS

Projekt realizowany w ramach koła naukowego Data Science

Autorzy: Ksawery Komasara, Mateusz Marcinkowski, Jakub Gronkiewicz

Celem projektu jest stworzenie narzędzia, które na podstawie zbioru danych (rezerwacji hotelowych) określi czy dana rezerwacja zostanie odwołana.

Poniżej opisano kluczowe wnioski, najważniejsze decyzje ze wszystkich etapów, wykresy, tabele porównujące jakość modeli.

## Eksproracyjna analiza danych

Zbiór analizowanych danych - Hotel_Reservations.csv -  można znaleźć w folderze Recources

### Informacje wyciągnięte na podstawie zbioru danych 

* Zmienna celu: booking_status
* Zmienna celu przyjmuje dwie wartości: Canceled / Not_canceled

* Typy danych: object, int64, float64

* Brak duplikatów

* Brak wartości pustych

* Wsytępują wartości odstające

* Do zastanowienia: Wykres korelacji zmiennych ze zmienną celu (Ważny wykres)

### Wnioski i spostrzeżenia

* Zmienną celu należy poddać mapowaniu

* Eliminacja wartości odstających

* Najważniejsze zmienne - avg_price_per_room / lead_time

* Wykres ze zobrazowaniem liczby wystąpień rezerwacji dla konkretnych cen pokojów hotelowych pokazał, że istnieje ponad 500 rezerwacji ze średnią ceną pokoju równą 0. Wynika to z dwóch czynników. Cena za pokój jest równa zero kiedy typ gościa hotelowego (zmienna market_segment_type) przyjumuje wartość "Complimetary", co oznacza, że rezerwacja nie jest typową rezerwacją i cena za pokój wynosi 0. Cena za pokój wyniesie 0 również gdy klient nie podał ani liczby nocy w tygodniu roboczym, ani liczby nocy weekendowych.

## Przetwarzanie danych

### Przygotowanie danych do treningu

* Mapowanie kategorycznej zmiennej celu: Canceled - 1, Not_canceled - 0

* Eliminacja cechy id rezerwacji - 'Booking_ID'

* Eliminacja outlierów w procesie preprocessingu

## Trening i ewaluacja modeli

Wykorzystano 5 modeli: regresję logistyczną, drzewo decyzyjne, random forest, adaBoost oraz XGBoost

### Regresja Logistyczna

* Trening modeeli - podział zbioru

* Tuning hiperparametrów - jakie paramtery, ich dobór, znaczenie parametrów

* Wybór metryk - jakie metryki wybrano i dlaczego

* Ocena jakości modelu - dane dosatrczone przez wykorzystane metryki

### Drzewo decyzjyne

* Trening modeeli - podział zbioru

* Tuning hiperparametrów - jakie paramtery, ich dobór, znaczenie parametrów

* Wybór metryk - jakie metryki wybrano i dlaczego

* Ocena jakości modelu - dane dosatrczone przez wykorzystane metryki

### RandomForest



### AdaBoost



### XGBoost



## Wybrany model

Istotność zmiennych oraz wnioski biznesowe


## Podsumowanie

Tu wskazanie na tabelkę, która porównuje wszystkie wykorzystane modele.
