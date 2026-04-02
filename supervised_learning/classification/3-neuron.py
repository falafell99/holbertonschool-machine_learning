#!/usr/bin/env python3
"""Module that defines a single neuron with cost calculation."""
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
        Updates and returns the private attribute __A.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Y: correct labels for the input data.
        A: activated output for each example.
        Returns the cost.
        """
        m = Y.shape[1]
        # Broken into multiple lines to stay under 79 characters
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
