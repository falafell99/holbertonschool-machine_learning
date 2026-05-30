#!/usr/bin/env python3
"""Module for Bayesian Optimization acquisition"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Class that performs Bayesian optimization on a noiseless 1D GP"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):  # noqa: E741
        """
        Constructor for the BayesianOptimization class.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement.
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_best = np.min(self.gp.Y)
            imp = Y_best - mu - self.xsi
        else:
            Y_best = np.max(self.gp.Y)
            imp = mu - Y_best - self.xsi

        # Инициализация массивов Z и EI нулями
        Z = np.zeros(sigma.shape)
        ei = np.zeros(sigma.shape)

        # Маска для избежания деления на ноль, если sigma равна 0
        mask = sigma > 0

        # Вычисление Z (стандартизированное улучшение)
        Z[mask] = imp[mask] / sigma[mask]

        # Вычисление EI
        ei[mask] = (imp[mask] * norm.cdf(Z[mask]) +
                    sigma[mask] * norm.pdf(Z[mask]))

        # Выбор точки с самым высоким EI
        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei
