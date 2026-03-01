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

* Częściej odwływano rezerwacje, w których rezerwowane miesjce parkingowe

* Częściej odwływano rezerwacje w sezonie wakacyjnym

* Znacznie częściej odwoływano rezerwacje gdy w ramach rezerwacji nie było specjalnych życzeń

* Im większy przedział czasu od rezerwacji do pobytu w hotelu tym wieksze prawdopodobieństwo odwołania rezerwacji

* Wykres ze zobrazowaniem liczby wystąpień rezerwacji dla konkretnych cen pokojów hotelowych pokazał, że istnieje ponad 500 rezerwacji ze średnią ceną pokoju równą 0. Wynika to z dwóch czynników. Cena za pokój jest równa zero kiedy typ gościa hotelowego (zmienna market_segment_type) przyjumuje wartość "Complimetary", co oznacza, że rezerwacja nie jest typową rezerwacją i cena za pokój wynosi 0. Cena za pokój wyniesie 0 również gdy klient nie podał ani liczby nocy w tygodniu roboczym, ani liczby nocy weekendowych.

## Przetwarzanie danych

### Przygotowanie danych do treningu

* Mapowanie kategorycznej zmiennej celu: Canceled - 1, Not_canceled - 0

* Eliminacja cechy id rezerwacji - 'Booking_ID'

* Eliminacja outlierów poprzez wyliczenie IQR oraz winsoryzację (na cechach avg_price_per_room, lead_time, no_of_week_nights, no_of_weekend_nights)

## Trening i ewaluacja modeli

Wykorzystano 5 modeli: regresję logistyczną, drzewo decyzyjne, random forest, adaBoost oraz XGBoost

### Regresja Logistyczna

* Trening modeli - zbiór testowy stanowi 10 % całego zbioru danych

* Tuning hiperparametrów - wykorzystanie GridSearchCV z solverami lbfgs / liblinear

* Wybór metryk

    * Confusion matrix
    * ROC AUC

* Ocena jakości modelu

    * Benachmark 80 %
    * Model 86 %
    * Model po HPO 88 %

### Drzewo decyzjyne

* Trening modeli - zbiór testowy stanowi 10 % całego zbioru danych

* Tuning hiperparametrów - wykorzystanie GridSearchCV z podziałąmi według gini i entropy

* Wybór metryk

    * Confusion matrix
    * ROC AUC

* Ocena jakości modelu

    * Benachmark 86 %
    * Model 84 % - overfitting
    * Model po HPO 88 %

### RandomForest



### AdaBoost



### XGBoost



## Wybrany model

Istotność zmiennych oraz wnioski biznesowe


## Podsumowanie

Tu wskazanie na tabelkę, która porównuje wszystkie wykorzystane modele.
