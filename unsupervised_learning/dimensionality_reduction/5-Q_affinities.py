#!/usr/bin/env python3
"""Module to calculate Q affinities in t-SNE"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities of the low-dimensional transformation Y.
    """
    # 1. Вычисляем квадрат попарных расстояний для матрицы Y
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    # 2. Вычисляем числитель (numerator) по формуле распределения Стьюдента
    num = 1 / (1 + D)

    # Расстояние от точки до самой себя не учитывается (зануляем диагональ)
    np.fill_diagonal(num, 0)

    # 3. Вычисляем матрицу Q, нормализуя числитель на общую сумму
    Q = num / np.sum(num)

    return Q, num
