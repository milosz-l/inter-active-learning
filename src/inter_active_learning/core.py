from data import split_active, fetch_and_prepare_titanic_data, fetch_and_prepare_mnist_data
from sampling import uncertainty_sampling, entropy_sampling, confidence_margin_sampling, confidence_quotient_sampling
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

RANDOM_STATE=1234
DATA_DICT = {
    "titanic": fetch_and_prepare_titanic_data,
    "mnist": fetch_and_prepare_mnist_data
}

def compute(args):
    return max(args, key=len)

def compute_metrics(pipe, X_valid, y_valid, data='titanic'):
    # Evaluate on validation set
    metric_dict = {}
    y_valid_pred = pipe.predict(X_valid)
    y_valid_prob = pipe.predict_proba(X_valid)

    metric_dict["Accuracy"] = accuracy_score(y_valid, y_valid_pred)
    metric_dict["Negative Log Loss"] = log_loss(y_valid, y_valid_prob)
    if data == 'titanic':
        y_valid_prob = y_valid_prob[:,1]
    metric_dict["AUC"] = roc_auc_score(y_valid, y_valid_prob, multi_class='ovr')
    return metric_dict

def active_learn(data: str,  stop_criterion, classifier, uncertainty_fc, data_splits: np.array = np.array([0.1, 0.7, 0.1, 0.1]), n_samples=100, random_state=RANDOM_STATE):
    """
    params:
        stop_criterion: function taking argument in form of metric dicts and returning True/False
    """
    if data in DATA_DICT.keys():
        X, y = DATA_DICT[data]()
    else:
        print(f'Available data sources: {DATA_DICT.keys()}')
        return None
    X_base, X_active, X_valid, X_test, y_base, y_active, y_valid, y_test = split_active(X, y, split=data_splits, random_state=random_state)
    iter = 0
    metric_dict = {
        "Accuracy": 0,
        "Negative Log Loss": np.Inf,
        "AUC": 0
    }
    while(not stop_criterion(metric_dict) and len(X_active) > 0):
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

def experiment(data: list, stop_criterion, classifiers: dict, uncertainty_fcs: dict, data_splits: np.array = np.array([0.1, 0.7, 0.1, 0.1]), n_samples=[100], random_state=RANDOM_STATE):
    results = pd.DataFrame(columns=['data', 'classifier', 'strategy', 'N sampled per iter', 'Dataset', 'Accuracy', 'Negative Log Loss', 'AUC', 'Iterations'])
    row = {'data': "", 'classifier': "", 'strategy': "", 'N sampled per iter': 0, 'Dataset': "", 'Accuracy': 0, 'Negative Log Loss': 0, 'AUC': 0, 'Iterations': 0}
    for d in data:
        row['data'] = d
        for c in classifiers.keys():
            row['classifier'] = c
            for u in uncertainty_fcs.keys():
                row['strategy'] = u
                for n in n_samples:
                    row['N sampled per iter'] = n
                    train, valid, test, iter = active_learn(d, stop_criterion, classifiers[c], uncertainty_fcs[u], data_splits=data_splits, n_samples=n, random_state=random_state)
                    row['Iterations'] = iter
                    for x, name in [(train, 'train'), (valid, 'valid'), (test, 'test')]:
                        row.update(x)
                        row['Dataset'] = name
                        results.loc[len(results)] = row
    return results

if __name__ == "__main__":
    # train, valid, test, iter = active_learn('titanic', lambda x: x['Accuracy'] > 0.9, GaussianNB(), uncertainty_sampling)
    # print(f"Iterations: {iter}")
    # print(train)
    # print(valid)
    # print(test)
    df = experiment(
            data=['mnist'],
            stop_criterion=lambda x: x['Accuracy'] > 0.9,
            classifiers={'KNN': KNeighborsClassifier(3)},
            uncertainty_fcs={"Uncertainty": uncertainty_sampling, "Entropy": entropy_sampling, "Confidence margin": confidence_margin_sampling, "Confidence quotient": confidence_quotient_sampling},
    )
    print(df)


