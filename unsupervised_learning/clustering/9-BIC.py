#!/usr/bin/env python3
"""Module to find the best number of clusters using BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters using Bayesian Information Criterion.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape
    results = []
    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, l_val = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None or m is None or S is None or l_val is None:
            return None, None, None, None

        # Calculate number of parameters (p)
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        # Calculate BIC
        bic_val = p * np.log(n) - 2 * l_val

        results.append((pi, m, S))
        log_likelihoods.append(l_val)
        bics.append(bic_val)

    log_likelihoods = np.array(log_likelihoods)
    bics = np.array(bics)

    best_idx = np.argmin(bics)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_likelihoods, bics
