#!/usr/bin/env python3
"""Module for Bayesian Optimization initialization"""
import numpy as np
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

        # Генерация ac_samples точек, равномерно распределенных между bounds
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
