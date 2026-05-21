#!/usr/bin/env python3
"""Module to calculate the cost of the t-SNE transformation"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation
    """
    # Защита от деления на 0 и вычисления логарифма от 0
    P_safe = np.maximum(P, 1e-12)
    Q_safe = np.maximum(Q, 1e-12)

    # Вычисление Дивергенции Кульбака-Лейблера
    C = np.sum(P * np.log(P_safe / Q_safe))

    return C
