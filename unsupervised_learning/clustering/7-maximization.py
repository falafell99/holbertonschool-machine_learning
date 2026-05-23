#!/usr/bin/env python3
"""Module to calculate the maximization step in the EM algorithm"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    # Validate that probabilities sum to 1 across clusters
    prob_sum = np.sum(g, axis=0)
    if not np.all(np.isclose(prob_sum, np.ones(X.shape[0]))):
        return None, None, None

    try:
        n, d = X.shape
        k = g.shape[0]

        # 1. Calculate the effective number of points in each cluster
        N_k = np.sum(g, axis=1)

        # 2. Update priors (pi)
        pi = N_k / n

        # 3. Update centroid means (m)
        m = np.matmul(g, X) / N_k[:, np.newaxis]

        # 4. Update covariance matrices (S) using exactly one loop
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            weighted_diff = g[i, :, np.newaxis] * diff
            S[i] = np.matmul(weighted_diff.T, diff) / N_k[i]

        return pi, m, S
    except Exception:
        return None, None, None
