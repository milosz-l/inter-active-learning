Richard Staszkiewicz
Miłosz Łopatto

# ZUM projekt - dokumentacja końcowa

## Temat projektu
10. Aktywne uczenie się modeli klasyfikacji na podstawie małych zbiorów trenujących przez zgłaszanie zapytania o prawdziwe wartości atrybutu docelowego dla ograniczonej liczby przykładów z dostarczonego dużego zbioru danych nieetykietowanych wybranych według określonych kryteriów (np. przykłady bliskie granicy decyzyjnej dotychczasowego modelu lub takie, dla których jego predykcje są obarczone największą niepewnością) i iteracyjne doskonalenie modelu na podstawie powiększanego w ten sposób zbioru trenującego. Implementacja w formie opakowania umożliwiającego użycie dowolnego algorytmu klasyfikacji dostępnego w środowisku R lub Python stosującego standardowy interfejs wywołania. Badanie wpływu użycia aktywnego uczenia się na jakość modeli klasyfikacji tworzonych na podstawie małych zbiorów trenujących za pomocą wybranych algorytmów dostępnych w środowisku R lub Python.

### Interpretacja tematu
Temat zinterpretowano jako polecenie zbudowania biblioteki w języku Python poddające obiekty o interfejsie klsyfikatora zaczerpniętym ze znanej biblioteki scikit-learn uczeniu aktywnemu na małej ilości otagowanych danych.

#### Pętla aktywnego uczenia
![Pętla aktywnego uczenia](docs/assets/active_learning.png)
źródło: [1]


## Opis części implementacyjnej
W ramach implementacji zostały zrealizowane osobno moduły do uczenia aktywnego oraz interfejsu użytkownika. W efekcie użytkownik może korzystać z biblioteki jak z dowolnej innej biblioteki, a dodatkowo będzie mógł alternatywnie wykorzystać interfejs użytkownika do łatwiejszego wykonywania eksperymentów.

### Lista dostępnych algorytmów klasyfikacji
<!---
(https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
--->
Zaimplementowany pakiet umożliwia wykonywanie eksperymentów z różnymi algorytmami klasyfikacji z biblioteki **SciKit Learn**:
- Nearest Neighbors (sklearn.neighbours.KNeighborsClassifier)
- Linear SVM (sklearn.SVM.SVC)
- RBF SVM (sklearn.SVM.SVC)
- Gaussian Process (sklearn.gaussian_process.GaussianProcessClassifier)
- Decision Tree (sklearn.tree.DecisionTreeClassifier)
- Random Forest (sklearn.ensemble.RandomForestClassifier)
- AdaBoost (sklearn.ensemble.AdaBoostClassifier)
- Naive Bayes (sklearn.naive_bayes.GaussianNB)
- QDA (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)
Wszelkie uzbierane dane numeryczne będące rezultatami działania poszczególnych algorytmów można przeanalizować zbiorowo jako DataFrame wygenerowany przy pomocy **Pandas**.

#### Parametry algorytmów
Algorytmy klasyfikujące nie są przedmiotem badania, które skupia się na active learningu i jego parametrze strategii. W związku z powyższym, oprócz tego jednego hiperparametru, reszta pozostanie stała. W większości przypadków zdecydowaliśmy się zostawić domyślne wartości hiperparametrów. Modele są inicjowane w następujący sposób:
```python
classifiers = {
    "KNN": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear", probability=True),
    "RBF SVM": SVC(kernel="rbf", probability=True),
    "Gaussian Process": GaussianProcessClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
}
```

### Funkcjonalność
Zaimplementowany pakiet umożliwia trenowanie algorytmów zgodnie z ideą aktywnego uczenia. Umożliwia do funkcja `active_learn` w pliku `core.py`. Możemy do niej przekazać następujące parametry:

- dane
- kryterium stopu jako funkcję opartą na jednej z metryk jakości, na przykład `lambda x: x["Accuracy"] > 0.9`
- model
- funkcja niepewności
- procentowy podział danych na zbiór treningowy, zbiór do aktywnego uczenia, zbiór walidacyjny oraz zbiór testowy
- liczba próbek dokładanych w jednej iteracji aktywnego uczenia
<!-- ^ TODO: czy to się zgadza? -->
- stan losowy umożliwiający reprodukcję otrzymanych wyników

Ponadto pakiet umożliwia uruchomienie całego eksperymentu porównawczego będącego na wyższym poziomie abstrakcji. Realizuje to funkcja `experiment` znajdująca się także w pliku `core.py`. Przyjmuje ona następujące parametry:

- lista dostępnych zbiorów danych na których mają zostać przeprowadzone eksperymenty
- kryterium stopu (identycznie jak we wcześniej opisywanej funkcji `active_learn`)
- lista modeli do przetestowania
- lista funkcji niepewności do przetestowania
- procentowy podział danych (identycznie jak we wcześniej opisywanej funkcji `active_learn`)
- lista liczb próbek dokładanych w jednej iteracji aktywnego uczenia
<!-- ^ TODO: czy to się zgadza? -->
- stan losowy umożliwiający reprodukcję otrzymanych wyników

Przykłady użycia:
TODO


### Charakterystyka zbiorów danych
Domyślnie pakiet umożliwia przeprowadzanie eksperymentów na poniższych zbiorach danych:

- Zbiór [Titanic](https://www.kaggle.com/datasets/brendan45774/test-file)
Będziemy dokonywać na nim klasyfikacji binarnej, dokładniej przewidywania wartości klasy `Survived`.
- Zbiór [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
W ramach biblioteki torchvision, zbiór ten jest reprezentowany poprzez 60000 kolorowych obrazków o wymiarach 32x32 podzielonych na 10 równolicznych klas.

### Preprocessing danych
Dla uproszczenia do klasyfikacji wykorzystane zostały jedynie kolumny numeryczne. Pominęliśmy kroki takie jak przykładowo one-hot encoding dla kolumn kategorycznych. Uznaliśmy, że główny temat i cel projektu tego nie wymaga, a takie podejście pozwoliło nam skrócić czas obliczeń.

Dane dzielone są na 4 zbiory: `Train`, `Active`, `Valid` oraz `Test`, gdzie:
- `Train` jest zbiorem danych do początkowego trenowania modeli
- `Active` jest zbiorem danych z którego brane są próbki w kolejnych iteracjach aktywnego uczenia
- `Valid` jest zbiorem walidacyjnym
- `Test` jest zbiorem testowym

### Opis interfejsu graficznego
Interfejs użytkownika podczas aktywnego uczenia został zrealizowany jako aplikacja webowa przy pomocy biblioteki **Streamlit**.

![Interfejs graficzny - przykładowy eksperyment](docs/assets/app1.png)

Interfejs graficzny pozwala na wygodne skonfigurowanie poniższych parametrów eksperymentu:

- wybór zbioru danych (na przykładzie wybrany został zbiór `Titanic`)
- wybór kryterium stopu (na przykładzie: `AUC`)
- ustalenie progu kryterium stopu na zbiorze walidacyjnym, po którym moduł przestaje wykonywać kolejne iteracje trenowania (na przykładzie: `0.85`)
- wybór modeli do przetestowania (na przykładzie wybrany został `Naive Bayes` oraz `Linear SVM`)
- wybór funkcji niepewności do przetestowania (na przykładzie wybrane zostały wszystkie strategie, czyli: `Uncertainty`, `Entropy`, `Confidence margin` oraz `Confidence quotient`)
- pierwszy suwak umożliwia zdefiniowanie podziału na dwie części - pierwsza część będzie dalej przeznaczona na zbiór treningowy oraz zbiór do aktywanego uczenia, a druga część będzie dalej przeznaczona na zbiór walidacyjny oraz zbiór treningowy (w tym wypadku 80% danych przeznaczamy na `Train+Active`, a 20% danych przeznaczamy na `Valid+Test`)
- następnie dwa suwaki umożliwiają dostosowanie proporcji zbiorów `Train+Active` oraz `Valid+Test` (w tym wypadku ostatecznie 10% przeznaczamy na zbiór `Train` i 70% na zbiór `Active`, a pozostałe 20% rozdzielamy po równo na zbiory `Valid` oraz `Test`)
- Ustawienie `Number of Samples per Iteration` pozwala nam zdefiniować liczbę przykładów dobieranych w kolejnych iteracjach aktywnego uczenia (w tym wypadku `10`)
- ostatnia opcja umożliwia nam wybranie zbiorów danych dla których wyniki mają być widoczne w wynikowej tabeli (w tym wypadku chcemy móc porównać wyniki dla zbiorów `Train` oraz `Valid`, natomiast nie chcemy znać wyników uzyskiwanych przez modele na zbiorze testowym)

Dodatkowo wygenerowane wyniki można w prosty sposób pobrać jako plik w formacie `.csv`:
![app2.png](docs/assets/app2.png)

TODO: filmik prezentujący interfejs graficzny


## Porównanie miar niepewności

Pakiet umożliwia realizowanie eksperymentów z wykorzystaniem następujących strategii zapytań:

![miary niepewności](docs/assets/miary_niepewnosci.png)
źródło: [2]

#### Entropia (Entropy sampling)

#### Najmniejsza ufność (Uncertainty sampling)

#### Margines ufności (Confidence margin sampling)

#### Iloraz ufności (Confidence quotient sampling)

### Przykładowy eksperyment - badanie wpływu strategii zapytań
Jako przykładowy eksperyment przeanalizujemy prezentowany wcześniej przykład.

![app1.py](/docs/assets/app1.png)

## Struktura projektu
TODO

## Pre-commit, autoformat, linter
TODO

## Źródła
[1]: https://github.com/modAL-python/modAL
[2]: http://elektron.elka.pw.edu.pl/~pcichosz/zum/slajdy/zum-s11.pdf
