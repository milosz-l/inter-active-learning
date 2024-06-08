"""
Active Learning Experimentation Module

This module provides functionality for performing active learning experiments on different datasets
using various classifiers and uncertainty sampling strategies.

Imports:
    - warnings: Standard library for warning control.
    - numpy as np: Numerical operations.
    - pandas as pd: Data manipulation and analysis.
    - sklearn.metrics: Metrics for evaluating model performance.
    - sklearn.neighbors: K-Nearest Neighbors classifier.
    - sklearn.pipeline: Pipeline creation.
    - sklearn.preprocessing: Data preprocessing.
    - .data: Custom module for data fetching and preparation.
    - .sampling: Custom module for uncertainty sampling strategies.

Functions:
    - compute_metrics: Computes accuracy, log loss, and AUC for the given model.
    - active_learn: Performs active learning using a specified classifier and uncertainty sampling strategy.
    - experiment: Runs multiple active learning experiments and collects results.

Constants:
    - RANDOM_STATE: Seed for random operations.
    - DATA_DICT: Dictionary mapping dataset names to their respective data fetching functions.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import fetch_and_prepare_mnist_data
from .data import fetch_and_prepare_titanic_data
from .data import split_active
from .sampling import confidence_margin_sampling
from .sampling import confidence_quotient_sampling
from .sampling import entropy_sampling
from .sampling import uncertainty_sampling

warnings.filterwarnings("ignore")

RANDOM_STATE = 1234
DATA_DICT = {"titanic": fetch_and_prepare_titanic_data, "mnist": fetch_and_prepare_mnist_data}


def compute_metrics(pipe, X_valid, y_valid, data="titanic"):
    """
    Computes evaluation metrics for a given model pipeline on the validation set.

    Args:
        pipe (Pipeline): The trained model pipeline.
        X_valid (np.array): Validation features.
        y_valid (np.array): Validation labels.
        data (str): Dataset name to determine specific metric calculations.

    Returns:
        dict: Dictionary containing accuracy, log loss, and AUC scores.
    """
    # Evaluate on validation set
    metric_dict = {}
    y_valid_pred = pipe.predict(X_valid)
    y_valid_prob = pipe.predict_proba(X_valid)

    metric_dict["Accuracy"] = accuracy_score(y_valid, y_valid_pred)
    metric_dict["Negative Log Loss"] = log_loss(y_valid, y_valid_prob)
    if data == "titanic":
        y_valid_prob = y_valid_prob[:, 1]
    metric_dict["AUC"] = roc_auc_score(y_valid, y_valid_prob, multi_class="ovr")
    return metric_dict


def active_learn(
    data: str,
    stop_criterion,
    classifier,
    uncertainty_fc,
    data_splits: np.array = np.array([0.1, 0.7, 0.1, 0.1]),
    n_samples=100,
    random_state=RANDOM_STATE,
):
    """
    Performs active learning using the specified classifier and uncertainty function.

    Args:
        data (str): Dataset name to be used.
        stop_criterion (function): Function that takes metric dicts and returns True/False to stop learning.
        classifier: Scikit-learn classifier.
        uncertainty_fc (function): Uncertainty sampling function.
        data_splits (np.array): Array defining data splits for training, active learning, validation, and testing.
        n_samples (int): Number of samples to query in each iteration.
        random_state (int): Seed for random operations.

    Returns:
        tuple: Training metrics, validation metrics, test metrics, and number of iterations.
    """
    if data in DATA_DICT.keys():
        X, y = DATA_DICT[data]()
    else:
        print(f"Available data sources: {DATA_DICT.keys()}")
        return None
    X_base, X_active, X_valid, X_test, y_base, y_active, y_valid, y_test = split_active(X, y, split=data_splits, random_state=random_state)
    iter = 0
    metric_dict = {"Accuracy": 0, "Negative Log Loss": np.Inf, "AUC": 0}
    while not stop_criterion(metric_dict) and len(X_active) > 0:
        pipe = make_pipeline(StandardScaler(), classifier)
        pipe.fit(X_base, y_base)

        metric_dict = compute_metrics(pipe, X_valid, y_valid, data)

        if stop_criterion(metric_dict):
            break

        # Select the most uncertain sample
        indices = uncertainty_fc(pipe, X_pool=X_active, y_pool=y_active, X_base=X_base, y_base=y_base, n_samples=n_samples)

        # Move the selected sample to the base set
        X_base = np.vstack([X_base, X_active[indices]])
        y_base = np.append(y_base, y_active[indices])

        X_active = np.delete(X_active, indices, axis=0)
        y_active = np.delete(y_active, indices, axis=0)

        iter += 1

    train_metrics = compute_metrics(pipe, X_base, y_base, data)
    test_metrics = compute_metrics(pipe, X_test, y_test, data)

    return train_metrics, metric_dict, test_metrics, iter


def experiment(
    data: list,
    stop_criterion,
    classifiers: dict,
    uncertainty_fcs: dict,
    data_splits: np.array = np.array([0.1, 0.7, 0.1, 0.1]),
    n_samples=[100],
    random_state=RANDOM_STATE,
):
    """
    Runs multiple active learning experiments and collects results.

    Args:
        data (list): List of datasets to be used.
        stop_criterion (function): Function that takes metric dicts and returns True/False to stop learning.
        classifiers (dict): Dictionary of classifiers to be used.
        uncertainty_fcs (dict): Dictionary of uncertainty sampling functions to be used.
        data_splits (np.array): Array defining data splits for training, active learning, validation, and testing.
        n_samples (list): List of sample sizes to query in each iteration.
        random_state (int): Seed for random operations.

    Returns:
        pd.DataFrame: DataFrame containing the results of the experiments.
    """
    results = pd.DataFrame(
        columns=[
            "data",
            "classifier",
            "strategy",
            "N sampled per iter",
            "Iterations",
            "train_Accuracy",
            "train_Negative Log Loss",
            "train_AUC",
            "valid_Accuracy",
            "valid_Negative Log Loss",
            "valid_AUC",
            "test_Accuracy",
            "test_Negative Log Loss",
            "test_AUC",
        ]
    )
    row = {
        "data": "",
        "classifier": "",
        "strategy": "",
        "N sampled per iter": 0,
        "Iterations": 0,
        "train_Accuracy": 0,
        "train_Negative Log Loss": 0,
        "train_AUC": 0,
        "valid_Accuracy": 0,
        "valid_Negative Log Loss": 0,
        "valid_AUC": 0,
        "test_Accuracy": 0,
        "test_Negative Log Loss": 0,
        "test_AUC": 0,
    }
    for d in data:
        row["data"] = d
        for c in classifiers.keys():
            row["classifier"] = c
            for u in uncertainty_fcs.keys():
                row["strategy"] = u
                for n in n_samples:
                    row["N sampled per iter"] = n
                    train, valid, test, iter = active_learn(
                        d,
                        stop_criterion,
                        classifiers[c],
                        uncertainty_fcs[u],
                        data_splits=data_splits,
                        n_samples=n,
                        random_state=random_state,
                    )
                    row["Iterations"] = iter
                    for metric in ["Accuracy", "Negative Log Loss", "AUC"]:
                        row[f"train_{metric}"] = train[metric]
                        row[f"valid_{metric}"] = valid[metric]
                        row[f"test_{metric}"] = test[metric]
                    results.loc[len(results)] = row.copy()
    return results


if __name__ == "__main__":
    # train, valid, test, iter = active_learn('titanic', lambda x: x['Accuracy'] > 0.9, GaussianNB(), uncertainty_sampling)
    # print(f"Iterations: {iter}")
    # print(train)
    # print(valid)
    # print(test)
    df = experiment(
        data=["titanic"],
        stop_criterion=lambda x: x["Accuracy"] > 0.9,
        classifiers={"KNN": KNeighborsClassifier(3)},
        uncertainty_fcs={
            "Uncertainty": uncertainty_sampling,
            "Entropy": entropy_sampling,
            "Confidence margin": confidence_margin_sampling,
            "Confidence quotient": confidence_quotient_sampling,
        },
    )
    df.to_csv("cos.txt")
