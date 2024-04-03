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
<!---
(https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
--->
W eksperymentach wykożystywane będę następujące algorytmy z biblioteki **SciKit Learn** w wariancie klasyfikatorów:
    * Nearest Neighbors (sklearn.neighbours.KNeighborsClassifier)
    * Linear SVM (sklearn.SVM.SVC)
    * RBF SVM (sklearn.SVM.SVC)
    * Gaussian Process (sklearn.gaussian_process.GaussianProcessClassifier)
    * Decision Tree (sklearn.tree.DecisionTreeClassifier)
    * Random Forest (sklearn.ensemble.RandomForestClassifier)
    * AdaBoost (sklearn.ensemble.AdaBoostClassifier)
    * Naive Bayes (sklearn.naive_bayes.GaussianNB)
    * QDA (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)
Wszelkie uzbierane dane numeryczne będące rezultatami działania algorytmu będą analizowane zbiorowo za pomocą funkcji biblioteki **Pandas**, głównie metod klasy pandas.DataFrame.
Interfejs webowy użytkownika podczas aktywnego uczenia zostanie zrealizowany za pomocą abstrakcji biblioteki **Streamlit**.

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
* Ze względu na brak działających modułów, nie stwierdzono jeszcze zasadności i stacku istnienia testów automatycznych.
