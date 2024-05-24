import numpy as np
def uncertainty_sampling(classifier, X_pool, n_samples=1):
    probs = classifier.predict_proba(X_pool)
    uncertainty = 1 - np.max(probs, axis=1)
    return np.argsort(uncertainty)[-min(n_samples, len(probs)):]