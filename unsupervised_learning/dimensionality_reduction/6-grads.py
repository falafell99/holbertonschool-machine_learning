#!/usr/bin/env python3
"""Module to calculate gradients of Y for t-SNE"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    # 1. Вычисляем матрицу весов (P_ij - Q_ij) * num_ij
    PQ_diff = P - Q
    W = PQ_diff * num

    # 2. Матричная оптимизация (трюк с Лапласианом)
    # Вместо циклов вычисляем сумму весов по строкам
    sum_W = np.sum(W, axis=1)

    # Создаем диагональную матрицу из этих сумм
    diag_W = np.diag(sum_W)

    # 3. Вычисляем итоговый градиент: dY = (diag_W - W) * Y
    dY = np.dot((diag_W - W), Y)

    return dY, Q
