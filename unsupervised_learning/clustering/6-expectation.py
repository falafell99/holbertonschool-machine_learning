#!/usr/bin/env python3
"""Module to calculate the expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1.0):
        return None, None

    try:
        g = np.zeros((k, n))

        # 1. Вычисляем числитель (совместную вероятность) для каждого кластера
        for j in range(k):
            P = pdf(X, m[j], S[j])
            if P is None:
                return None, None
            g[j] = pi[j] * P

        # 2. Вычисляем знаменатель (маржинальную вероятность)
        marginal = np.sum(g, axis=0)

        # 3. Вычисляем логарифмическое правдоподобие (Log-Likelihood)
        # Заменяем опасную переменную 'l' на 'log_likelihood'
        log_likelihood = np.sum(np.log(marginal))

        # 4. Вычисляем итоговые апостериорные вероятности (Posterior)
        g = g / marginal

        return g, log_likelihood
    except Exception:
        return None, None
