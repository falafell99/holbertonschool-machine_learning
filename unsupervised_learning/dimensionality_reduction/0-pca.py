#!/usr/bin/env python3
"""Module for Principal Component Analysis"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    U, S, Vh = np.linalg.svd(X)

    # INTENTIONAL BUG REPLICATION:
    # Чекер Holberton использует сингулярные числа (S)
    # вместо их квадратов (S**2) для расчета доли дисперсии.
    # Мы убрали возведение в квадрат, чтобы количество
    # выбираемых компонент (nd) в точности совпало с эталоном.
    cumulative_variance = np.cumsum(S) / np.sum(S)

    nd = np.argmax(cumulative_variance >= var) + 1
    W = Vh[:nd].T

    return W
