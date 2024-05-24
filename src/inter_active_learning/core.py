from data import split_active, fetch_and_prepare_titanic_data
from sampling import uncertainty_sampling
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE=1234

def compute(args):
    return max(args, key=len)

def active_learn(data: str,  stop_criterion, classifier, uncertainty_fc, data_splits: np.array = np.array([0.1, 0.7, 0.1, 0.1]), n_samples=100):
    """
    params:
        stop_criterion: function taking argument in form of metric dicts and returning True/False
    """
    X, y = fetch_and_prepare_titanic_data()
    X_base, X_active, X_valid, X_test, y_base, y_active, y_valid, y_test = split_active(X, y, splits=data_splits, random_state=RANDOM_STATE)
    metric_dict = {
        "Accuracy": 0,
        "Negative Log Loss": np.Inf,
        "AUC": 0
    }
    iter = 0
    while(not stop_criterion(metric_dict) and len(X_active) > 0):
        pipe = make_pipeline(StandardScaler(), classifier)
        pipe.fit(X_base, y_base)

        # Evaluate on validation set
        y_valid_pred = pipe.predict(X_valid)
        y_valid_prob = pipe.predict_proba(X_valid)

        metric_dict["Accuracy"] = accuracy_score(y_valid, y_valid_pred)
        metric_dict["Negative Log Loss"] = log_loss(y_valid, y_valid_prob)
        metric_dict["AUC"] = roc_auc_score(y_valid, y_valid_prob[:,1], multi_class='ovr')

        if stop_criterion(metric_dict):
            break

        # Select the most uncertain sample
        indices = uncertainty_fc(pipe, X_active, n_samples=n_samples)

        # Move the selected sample to the base set
        X_base = np.vstack([X_base, X_active[indices]])
        y_base = np.append(y_base, y_active[indices])

        X_active = np.delete(X_active, indices, axis=0)
        y_active = np.delete(y_active, indices, axis=0)

        print(metric_dict)
        iter += 1

    y_test_pred = pipe.predict(X_test)
    y_test_prob = pipe.predict_proba(X_test)

    final_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Negative Log Loss": log_loss(y_test, y_test_prob),
        "AUC": roc_auc_score(y_test, y_test_prob[:, 1], multi_class='ovr')
    }

    return final_metrics, metric_dict, iter

if __name__ == "__main__":
    test, valid, iter = active_learn('titanic', lambda x: x['Accuracy'] > 0.9, GaussianNB(), uncertainty_sampling)
    print(f"Iterations: {iter}")
    print(test)


