import numpy as np
from scipy.stats import entropy

def uncertainty_sampling(classifier, X_pool, n_samples=1, **kwargs):
    probs = classifier.predict_proba(X_pool)
    uncertainty = 1 - np.max(probs, axis=1)
    return np.argsort(uncertainty)[-min(n_samples, len(probs)):]

def entropy_sampling(classifier, X_pool, n_samples=1, **kwargs):
    probs = classifier.predict_proba(X_pool)
    entropy_values = np.apply_along_axis(entropy, 1, probs)
    return np.argsort(entropy_values)[-min(n_samples, len(entropy_values)):]

def confidence_margin_sampling(classifier, X_pool, n_samples=1, **kwargs):
    probs = classifier.predict_proba(X_pool)
    sorted_probs = np.sort(probs, axis=1)
    confidence_margins = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
    return np.argsort(confidence_margins)[-min(n_samples, len(confidence_margins)):]

def confidence_quotient_sampling(classifier, X_pool, n_samples=1, **kwargs):
    probs = classifier.predict_proba(X_pool)
    sorted_probs = np.sort(probs, axis=1)
    confidence_quotients = sorted_probs[:, -2] / sorted_probs[:, -1]
    return np.argsort(confidence_quotients)[-min(n_samples, len(confidence_quotients)):]
