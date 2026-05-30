#!/usr/bin/env python3
"""Module for Gaussian Process prediction"""
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

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a GP.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y)
        mu = mu.reshape(-1)

        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diagonal(cov)

        return mu, sigma
