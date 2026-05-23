#!/usr/bin/env python3
"""Module to calculate the PDF of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    try:
        # Вычисляем разницу X - m (broadcasting automatically aligns m to X)
        diff = X - m

        # Находим обратную матрицу и детерминант ковариации S
        S_inv = np.linalg.inv(S)
        S_det = np.linalg.det(S)

        # Вычисляем коэффициент перед экспонентой
        coeff = 1.0 / np.sqrt(((2 * np.pi) ** d) * S_det)

        # Вычисляем показатель степени: -0.5 * (X - m)^T * S_inv * (X - m)
        # Вместо огромного матричного умножения и извлечения диагонали,
        # мы умножаем матрицу на вектор, а затем делаем поэлементное
        # умножение и сумму по строкам (axis=1). Это заменяет np.diag!
        exponent = -0.5 * np.sum(np.matmul(diff, S_inv) * diff, axis=1)

        # Собираем итоговое значение PDF
        P = coeff * np.exp(exponent)

        # Защита от слишком малых значений (underflow)
        P = np.maximum(P, 1e-300)

        return P
    except Exception:
        return None
