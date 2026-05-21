#!/usr/bin/env python3
"""Module to calculate symmetric P affinities of a data set"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set
    """
    n, d = X.shape
    D, P, betas, H_target = P_init(X, perplexity)

    for i in range(n):
        # Исключаем расстояние от точки до самой себя (диагональ)
        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        beta = betas[i, 0]
        beta_min = None
        beta_max = None

        # Вычисляем начальную энтропию
        Hi, Pi = HP(Di, beta)
        H_diff = Hi - H_target

        # Бинарный поиск для нахождения правильного значения beta
        while np.abs(H_diff) > tol:
            if H_diff > 0:
                # Энтропия слишком высока -> разброс слишком велик -> нужно сузить радиус
                # Увеличение beta уменьшает радиус (дисперсию)
                beta_min = beta
                if beta_max is None:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                # Энтропия слишком низка -> разброс слишком мал -> нужно расширить радиус
                # Уменьшение beta увеличивает радиус
                beta_max = beta
                if beta_min is None:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

            # Пересчитываем с новым beta
            Hi, Pi = HP(Di, beta)
            H_diff = Hi - H_target

        # Сохраняем найденное значение beta (опционально, но полезно для понимания)
        betas[i, 0] = beta

        # Вставляем вероятности обратно в матрицу P (пропуская диагональ)
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    # Вычисление симметричных вероятностей по формуле t-SNE: (P + P.T) / (2 * n)
    P = (P + P.T) / (2 * n)

    return P
