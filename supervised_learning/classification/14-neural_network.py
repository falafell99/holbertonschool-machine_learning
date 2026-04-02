#!/usr/bin/env python3
"""Module defining a neural network with training cycle."""
import numpy as np


class NeuralNetwork:
    """Class representing a neural network."""

    def __init__(self, nx, nodes):
        """
        Initializes neural network.
        nx: number of input features.
        nodes: number of nodes in hidden layer.
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
        """Getter __W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter __b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter __A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter __W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter __b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter __A2."""
        return self.__A2

    def forward_prop(self, X):
        """Calculates neural network forward propagation."""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates model cost."""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions of neural network."""
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent."""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.dot(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.
        iterations: number of iterations.
        alpha: learning rate.
        Returns evaluation of training data after iterations.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
