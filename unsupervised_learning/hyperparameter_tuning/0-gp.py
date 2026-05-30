#!/usr/bin/env python3
"""Module for Gaussian Process initialization"""
import numpy as np


class GaussianProcess:
    """Class that represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):  # noqa: E741
        """
        Constructor for the GaussianProcess class.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l  # noqa: E741
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).
        """
        sqdist = (X1 - X2.T) ** 2
        K = (self.sigma_f ** 2) * np.exp(-sqdist / (2 * (self.l ** 2)))
        return K
