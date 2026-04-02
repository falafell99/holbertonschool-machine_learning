#!/usr/bin/env python3
"""Module that defines a single neuron with forward propagation."""
import numpy as np


class Neuron:
    """Class representing a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initializes the neuron.
        nx is the number of input features.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector __W."""
        return self.__W

    @property
    def b(self):
        """Getter for the bias __b."""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output __A."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        X is a numpy.ndarray with shape (nx, m) containing input data.
        Updates and returns the private attribute __A.
        """
        # Z = W * X + b
        Z = np.dot(self.__W, X) + self.__b
        # Sigmoid activation: A = 1 / (1 + exp(-Z))
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
