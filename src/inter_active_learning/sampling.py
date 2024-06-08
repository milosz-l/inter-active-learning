"""
Uncertainty Sampling Strategies Module

This module provides various uncertainty sampling strategies for active learning. Each function selects the most uncertain samples from a given pool based on different criteria.

Imports:
    - numpy as np: Numerical operations.
    - scipy.stats.entropy: Entropy calculation.

Functions:
    - uncertainty_sampling: Selects samples with the highest uncertainty based on prediction probabilities.
    - entropy_sampling: Selects samples with the highest entropy.
    - confidence_margin_sampling: Selects samples with the smallest confidence margin between the top two predicted probabilities.
    - confidence_quotient_sampling: Selects samples with the highest confidence quotient between the top two predicted probabilities.
"""

import numpy as np
from scipy.stats import entropy


def uncertainty_sampling(classifier, X_pool, n_samples=1, **kwargs):
    """
    Selects samples with the highest uncertainty based on prediction probabilities.

    Args:
        classifier: The classifier with a predict_proba method.
        X_pool (np.array): The pool of samples to select from.
        n_samples (int): The number of samples to select.
        **kwargs: Additional arguments.

    Returns:
        np.array: Indices of the selected samples.
    """
    probs = classifier.predict_proba(X_pool)
    uncertainty = 1 - np.max(probs, axis=1)
    return np.argsort(uncertainty)[-min(n_samples, len(probs)) :]


def entropy_sampling(classifier, X_pool, n_samples=1, **kwargs):
    """
    Selects samples with the highest entropy.

    Args:
        classifier: The classifier with a predict_proba method.
        X_pool (np.array): The pool of samples to select from.
        n_samples (int): The number of samples to select.
        **kwargs: Additional arguments.

    Returns:
        np.array: Indices of the selected samples.
    """
    probs = classifier.predict_proba(X_pool)
    entropy_values = np.apply_along_axis(entropy, 1, probs)
    return np.argsort(entropy_values)[-min(n_samples, len(entropy_values)) :]


def confidence_margin_sampling(classifier, X_pool, n_samples=1, **kwargs):
    """
    Selects samples with the smallest confidence margin between the top two predicted probabilities.

    Args:
        classifier: The classifier with a predict_proba method.
        X_pool (np.array): The pool of samples to select from.
        n_samples (int): The number of samples to select.
        **kwargs: Additional arguments.

    Returns:
        np.array: Indices of the selected samples.
    """
    probs = classifier.predict_proba(X_pool)
    sorted_probs = np.sort(probs, axis=1)
    confidence_margins = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
    return np.argsort(confidence_margins)[-min(n_samples, len(confidence_margins)) :]


def confidence_quotient_sampling(classifier, X_pool, n_samples=1, **kwargs):
    """
    Selects samples with the highest confidence quotient between the top two predicted probabilities.

    Args:
        classifier: The classifier with a predict_proba method.
        X_pool (np.array): The pool of samples to select from.
        n_samples (int): The number of samples to select.
        **kwargs: Additional arguments.

    Returns:
        np.array: Indices of the selected samples.
    """
    probs = classifier.predict_proba(X_pool)
    sorted_probs = np.sort(probs, axis=1)
    confidence_quotients = sorted_probs[:, -2] / sorted_probs[:, -1]
    return np.argsort(confidence_quotients)[-min(n_samples, len(confidence_quotients)) :]
