#!/usr/bin/env python3
"""Module to calculate Shannon entropy and P affinities in t-SNE"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    """
    # Вычисляем числитель формулы t-SNE: экспоненту от отрицательного
    # квадрата расстояния, умноженного на параметр бета.
    Pi_num = np.exp(-Di * beta)

    # Находим сумму всех числителей для нормализации
    sum_Pi = np.sum(Pi_num)

    # Нормализуем, чтобы сумма всех P была равна 1 (получаем вероятности)
    Pi = Pi_num / sum_Pi

    # Вычисляем Энтропию Шеннона по основанию 2.
    # np.maximum используется для предотвращения ошибки log2(0),
    # если вдруг Pi окажется слишком маленьким нулем для float64.
    Hi = -np.sum(Pi * np.log2(np.maximum(Pi, 1e-300)))

    return Hi, Pi
