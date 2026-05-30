#!/usr/bin/env python3
"""Module for Bayesian Optimization optimization"""
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

        Z = np.zeros(sigma.shape)
        ei = np.zeros(sigma.shape)
        mask = sigma > 0

        Z[mask] = imp[mask] / sigma[mask]

        term1 = imp[mask] * norm.cdf(Z[mask])
        term2 = sigma[mask] * norm.pdf(Z[mask])
        ei[mask] = term1 + term2

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.
        """
        for _ in range(iterations):
            # 1. Находим самую перспективную точку
            X_next, _ = self.acquisition()

            # 2. Условие ранней остановки:
            # Если алгоритм предложил точку, которую мы УЖЕ проверяли,
            # значит мы сошлись к глобальному оптимуму (или застряли).
            if X_next in self.gp.X:
                break

            # 3. Вычисляем реальное значение нашей функции (Обучаем модель)
            Y_next = self.f(X_next)

            # 4. Обновляем Гауссовский процесс новыми знаниями
            self.gp.update(X_next, Y_next)

        # После завершения цикла (или ранней остановки) находим лучший результат
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
