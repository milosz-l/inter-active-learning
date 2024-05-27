from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataset(data: str, balace=0.5):
    if data.lower()=="titanic":
        return fetch_openml(name='titanic', version=1)
    elif data.lower()=="cifar-10":
        return fetch_openml(name='cifar-10')
    else:
        raise NotImplementedError("Classification for given dataset is not yet implemented")

def fetch_and_prepare_titanic_data():
    # Fetch the Titanic dataset
    titanic = fetch_openml(name='titanic', version=1, as_frame=True)
    df = titanic.frame

    # Select relevant columns
    columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']
    df = df[columns]

    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    #df.dropna(subset='embarked', inplace=True)

    # Convert categorical column 'Sex' to numerical
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df = df.dropna()

    # Split features and target
    X = df.drop('survived', axis=1)
    y = df['survived'].astype(int)

    return X.to_numpy(), y.to_numpy()

def fetch_and_prepare_mnist_data():
    mnist = fetch_openml(name='mnist_784', as_frame=True)
    df = mnist.frame
    X = df.drop('class', axis=1)
    y = df['class'].astype(int)

    return X.to_numpy(), y.to_numpy()

def split_active(X, y, splits: np.array = np.array([0.1, 0.7, 0.1, 0.1]), random_state=1234):
    Xes = []
    yes = []
    splits *= len(X)
    while len(splits) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(splits[0]), random_state=random_state)
        Xes.append(X_test)
        yes.append(y_test)
        X = X_train
        y = y_train
        splits = np.delete(splits, 0)
    Xes.append(X)
    yes.append(y)
    return Xes + yes

if __name__ == "__main__":
    # ans = split_active(np.array([1] * 5000), np.array([2] * 5000), np.array([0.1, 0.7, 0.1, 0.1]))
    # print([len(x) for x in ans])

    print(fetch_and_prepare_mnist_data()[1].shape)