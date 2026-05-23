#!/usr/bin/env python3
"""Module to optimize K for K-means"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmin) is not int or kmin <= 0:
        return None, None
    if type(kmax) is not int or kmax <= 0 or kmax <= kmin:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    first_var = None

    for k in range(kmin, kmax + 1):
        # Запускаем K-means для текущего k
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        # Вычисляем дисперсию
        var = variance(X, C)

        # Запоминаем дисперсию самого первого запуска (kmin)
        if k == kmin:
            first_var = var

        # Считаем разницу между базовой дисперсией и текущей
        d_vars.append(first_var - var)

    return results, d_vars
