"""
Data Preparation Module

This module provides functionality for fetching, preparing, and splitting datasets for machine learning tasks.

Imports:
    - fetch_openml: Function from sklearn.datasets to fetch datasets from OpenML.
    - train_test_split: Function from sklearn.model_selection to split datasets.
    - numpy as np: Numerical operations.

Functions:
    - get_dataset: Fetches the specified dataset from OpenML.
    - fetch_and_prepare_titanic_data: Fetches and prepares the Titanic dataset.
    - fetch_and_prepare_mnist_data: Fetches and prepares the MNIST dataset.
    - split_active: Splits the dataset into training, active learning, validation, and test sets.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np


def get_dataset(data: str, balace=0.5):
    """
    Fetches the specified dataset from OpenML.

    Args:
        data (str): The name of the dataset to fetch. Options are "titanic" and "cifar-10".
        balance (float): Proportion of classes to balance (currently unused).

    Returns:
        Bunch: The fetched dataset.

    Raises:
        NotImplementedError: If the dataset is not implemented.
    """
    if data.lower() == "titanic":
        return fetch_openml(name="titanic", version=1)
    elif data.lower() == "cifar-10":
        return fetch_openml(name="cifar-10")
    else:
        raise NotImplementedError("Classification for given dataset is not yet implemented")


def fetch_and_prepare_titanic_data():
    """
    Fetches and prepares the Titanic dataset for machine learning tasks.

    This function:
        - Fetches the Titanic dataset from OpenML.
        - Selects relevant columns.
        - Handles missing values.
        - Converts categorical columns to numerical.
        - Splits the data into features and target.

    Returns:
        tuple: A tuple containing:
            - X (np.array): The feature matrix.
            - y (np.array): The target vector.
    """
    # Fetch the Titanic dataset
    titanic = fetch_openml(name="titanic", version=1, as_frame=True)
    df = titanic.frame

    # Select relevant columns
    columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "survived"]
    df = df[columns]

    # Handle missing values
    df["age"].fillna(df["age"].median(), inplace=True)
    # df.dropna(subset='embarked', inplace=True)

    # Convert categorical column 'Sex' to numerical
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df = df.dropna()

    # Split features and target
    X = df.drop("survived", axis=1)
    y = df["survived"].astype(int)

    return X.to_numpy(), y.to_numpy()


def fetch_and_prepare_mnist_data():
    """
    Fetches and prepares the MNIST dataset for machine learning tasks.

    This function:
        - Fetches the MNIST dataset from OpenML.
        - Splits the data into features and target.

    Returns:
        tuple: A tuple containing:
            - X (np.array): The feature matrix.
            - y (np.array): The target vector.
    """
    mnist = fetch_openml(name="mnist_784", as_frame=True)
    df = mnist.frame
    X = df.drop("class", axis=1)
    y = df["class"].astype(int)

    return X.to_numpy(), y.to_numpy()


def split_active(X, y, split: np.array = np.array([0.1, 0.7, 0.1, 0.1]), random_state=1234):
    """
    Splits the dataset into training, active learning, validation, and test sets (potentially more).

    Args:
        X (np.array): The feature matrix.
        y (np.array): The target vector.
        split (np.array): Array defining the proportion of data for each split (training, active, validation, test by default).
        random_state (int): Seed for random operations.

    Returns:
        list: A list containing the split datasets in the order X then corresponding Y. By default:
            - X_base: Training features.
            - X_active: Active learning features.
            - X_valid: Validation features.
            - X_test: Test features.
            - y_base: Training targets.
            - y_active: Active learning targets.
            - y_valid: Validation targets.
            - y_test: Test targets.
    """
    Xes = []
    yes = []
    splits = split * len(X)
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
