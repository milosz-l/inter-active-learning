import sklearn
from sklearn.datasets import fetch_openml
from .data import split_active
import numpy as np

RANDOM_STATE=1234

def compute(args):
    return max(args, key=len)

def active_learn(data: str,  stop_criterion: float, classifier, uncertainty_metric, data_splits: np.array = np.array([0.1, 0.7, 0.1, 0.1])):
    X, y = fetch_openml(data, version=0, return_X_y=True)
    X_base, X_active, X_valid, X_test, y_base, y_active, y_valid, y_test = split_active(X, y, splits=data_splits, random_state=RANDOM_STATE)



