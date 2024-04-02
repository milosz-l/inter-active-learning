Richard Staszkiewicz
Miłosz Łopatto

# ZUM projekt - dokumentacja wstępna

## Temat projektu
10. Aktywne uczenie się modeli klasyfikacji na podstawie małych zbiorów trenujących przez zgłaszanie zapytania o prawdziwe wartości atrybutu docelowego dla ograniczonej liczby przykładów z dostarczonego dużego zbioru danych nieetykietowanych wybranych według określonych kryteriów (np. przykłady bliskie granicy decyzyjnej dotychczasowego modelu lub takie, dla których jego predykcje są obarczone największą niepewnością) i iteracyjne doskonalenie modelu na podstawie powiększanego w ten sposób zbioru trenującego. Implementacja w formie opakowania umożliwiającego użycie dowolnego algorytmu klasyfikacji dostępnego w środowisku R lub Python stosującego standardowy interfejs wywołania. Badanie wpływu użycia aktywnego uczenia się na jakość modeli klasyfikacji tworzonych na podstawie małych zbiorów trenujących za pomocą wybranych algorytmów dostępnych w środowisku R lub Python.

## Interpretacja tematu
Temat zinterpretowano jako polecenie zbudowania biblioteki w języku Python poddające obiekty o interfejsie klsyfikatora zaczerpniętym ze znanej biblioteki scikit-learn uczeniu aktywnemu na małej ilości otagowanych danych.

## Opis części implementacyjnej
W ramach implementacji zostaną zrealizowane osobno moduły do algorytmu uczenia aktywnego, interfejsu oraz integracji.

### Lista wykorzystanych w eksperymentach algorytmów
TODO: listę algorytmów, które będą wykorzystane w eksperymentach (ze wskazaniem wykorzystywanych bibliotek, i klas/funkcji)
-> Scikit Learn (algosy)
-> Pandas (dane)
-> Streamlit (? frontend?)

## Plan badań
TODO

### Cel poszczególnych badań
TODO

### Charakterystyka zbiorów danych
TODO

### Badane parametry algorytmów
TODO?

### Miary jakości i procedury oceny modeli
TODO

## Otwarte kwestie wymagające późniejszego rozwiązania (wraz z wyjaśnieniem powodów, dla których ich rozwiązanie jest odłożone na później)
