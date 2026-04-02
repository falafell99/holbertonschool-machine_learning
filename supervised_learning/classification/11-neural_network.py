#!/usr/bin/env python3
"""Module that defines a neural network with cost calculation."""
import numpy as np


class NeuralNetwork:
    """Class representing a neural network for binary classification."""

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.
        nx: number of input features.
        nodes: number of nodes in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for __W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for __b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for __A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for __W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for __b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for __A2."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        Updates and returns __A1 and __A2.
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Y: correct labels.
        A: activated output (prediction).
        Returns the cost.
        """
        m = Y.shape[1]
        # Binary Cross-Entropy formula formatted for PEP 8
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
