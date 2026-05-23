#!/usr/bin/env python3
"""Module to initialize variables of a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables of a Gaussian Mixture Model.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape

    # pi: Priors of each cluster, initialized evenly
    pi = np.full(shape=(k,), fill_value=1/k)

    # m: Centroid means initialized with K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # S: Covariance matrices initialized as identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
