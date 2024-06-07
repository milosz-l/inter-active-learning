Richard Staszkiewicz
Miłosz Łopatto

# ZUM projekt - dokumentacja końcowa

## Temat projektu
10. Aktywne uczenie się modeli klasyfikacji na podstawie małych zbiorów trenujących przez zgłaszanie zapytania o prawdziwe wartości atrybutu docelowego dla ograniczonej liczby przykładów z dostarczonego dużego zbioru danych nieetykietowanych wybranych według określonych kryteriów (np. przykłady bliskie granicy decyzyjnej dotychczasowego modelu lub takie, dla których jego predykcje są obarczone największą niepewnością) i iteracyjne doskonalenie modelu na podstawie powiększanego w ten sposób zbioru trenującego. Implementacja w formie opakowania umożliwiającego użycie dowolnego algorytmu klasyfikacji dostępnego w środowisku R lub Python stosującego standardowy interfejs wywołania. Badanie wpływu użycia aktywnego uczenia się na jakość modeli klasyfikacji tworzonych na podstawie małych zbiorów trenujących za pomocą wybranych algorytmów dostępnych w środowisku R lub Python.

### Interpretacja tematu
Temat zinterpretowano jako polecenie zbudowania biblioteki w języku Python poddające obiekty o interfejsie klsyfikatora zaczerpniętym ze znanej biblioteki scikit-learn uczeniu aktywnemu na małej ilości otagowanych danych.

## Opis części implementacyjnej
W ramach implementacji zostały zrealizowane osobno moduły do uczenia aktywnego oraz interfejsu użytkownika. W efekcie użytkownik może korzystać z biblioteki jak z dowolnej innej biblioteki, a dodatkowo będzie mógł alternatywnie wykorzystać interfejs użytkownika do łatwiejszego wykonywania eksperymentów.

### Struktura projektu
TODO

### Pre-commit, autoformat, linter
TODO

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

### Opis zaimplementowanego pakietu
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

def experiment(
    data: list,
    stop_criterion,
    classifiers: dict,
    uncertainty_fcs: dict,
    data_splits: np.array = np.array([0.1, 0.7, 0.1, 0.1]),
    n_samples=[100],
    random_state=RANDOM_STATE,
):

#### Opis interfejsu graficznego
Interfejs użytkownika podczas aktywnego uczenia został zrealizowany jako aplikacja webowa przy pomocy biblioteki **Streamlit**.



## Opis części badawczej
W ramach części badawczej przeprowadzona została przykładowa analiza porównawcza strategii zapytań dla wybranych algorytmów klasyfikacji na zbiorze danych Titanic.

### Preprocessing danych
Dla uproszczenia do klasyfikacji wykorzystane zostały jedynie kolumny numeryczne. Pominęliśmy kroki takie jak przykładowo one-hot encoding dla kolumn kategorycznych. Uznaliśmy, że główny temat i cel projektu tego nie wymaga, a takie podejście pozwoliło nam skrócić czas obliczeń.

### Badanie strategii zapytań


### Cel poszczególnych badań
- Badania Integracyjne
Są to badania zaprojektowane by naśladowały testy integracyjne systemu z symulowanym użytkownikiem. Nie będą się one wiązały z wynikami odrębnymi od informacji systemowych i będą prowadzone wyłącznie na Etapie Implementacji w celu weryfikacji poprawnego działania systemu.

- Badania Wpływu Strategii Zapytań
Są to badania wpływu różnych strategii wyznacznia punktów do otagowania przez algorytm aktywnego uczenia. Zamierzone jest zbadanie co najmniej 3 strategii: ATS, Uncertainty Sampling oraz Expected Error Reduction.

### Charakterystyka zbiorów danych
Planowanym jest używanie dwóch rodzajów zbiorów:

- Zbiór [Titanic](https://www.kaggle.com/datasets/brendan45774/test-file)
Będziemy dokonywać na nim klasyfikacji binarnej, dokładniej przewidywania wartości klasy `Survived`. Ze względu na prostą możliwość zbudowania automatycznej "wyroczni" niezbędnej w algortymie uczenia aktywnego, na tym zbiorze przeprowadzane będą eksperymenty numeryczne wpływu strategii próbkowania na budowę modelu.
- Zbiór [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
W ramach biblioteki torchvision, zbiór ten jest reprezentowany poprzez 60000 kolorowych obrazków o wymiarach 32x32 podzielonych na 10 równolicznych klas. Ze względu na jego prostą interpretowalność przez człowieka będzie on wykorzystywany jako zbiór testowy w interaktywnym interfejsie graficznym w Streamlit.

### Badane parametry algorytmów
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

### Miary jakości i procedury oceny modeli
- Badania Integracyjne
Ocenie będzie poddawany przyrost informacyjny dot. stanu kodu oraz czas odpowiedzi systemu. Ocena nastąpi poprzez skonfrontowanie rezultatów ze spodziewanymi.


- Badanie Wpływu Strategii Zapytań
Strategie zapytań będą mierzone na każdym jednostkowym przyroście informacyjnym poprzez ocenę każdego z modelów miarami Accuracy, Negative Log Loss oraz ROC Area Under Curve. Wspólnie tendencje tych trzech metryk będą poddawane analizie porównawczej na późniejszych etapach oceny wpływu parametru selekcji.

## Otwarte kwestie wymagające późniejszego rozwiązania (wraz z wyjaśnieniem powodów, dla których ich rozwiązanie jest odłożone na później)
- Ze względu na brak działających modułów, nie stwierdzono jeszcze zasadności i stacku istnienia testów automatycznych.
