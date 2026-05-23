#!/usr/bin/env python3
"""Module to initialize variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape

    # 1. pi: Равномерное начальное распределение вероятностей для каждого кластера
    pi = np.full(shape=(k,), fill_value=1/k)

    # 2. m: Центроиды инициализируются с помощью алгоритма K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # 3. S: Ковариационные матрицы инициализируются как единичные матрицы
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
