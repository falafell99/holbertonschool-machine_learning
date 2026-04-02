#!/usr/bin/env python3
"""Module defining a deep neural network with a training cycle."""
import numpy as np


class DeepNeuralNetwork:
    """Class representing a deep neural network."""

    def __init__(self, nx, layers):
        """Initializes deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if i == 0 else layers[i - 1]

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            self.__weights[w_key] = np.random.randn(layers[i], prev_nodes) * \
                np.sqrt(2 / prev_nodes)
            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for __L."""
        return self.__L

    @property
    def cache(self):
        """Getter for __cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for __weights."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates deep neural network forward propagation."""
        self.__cache["A0"] = X
        for i in range(self.__L):
            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            a_prev_key = "A" + str(i)
            a_curr_key = "A" + str(i + 1)

            Z = np.dot(self.__weights[w_key], self.__cache[a_prev_key]) + \
                self.__weights[b_key]
            self.__cache[a_curr_key] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model."""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions of deep neural network."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            w_key = "W" + str(i)
            b_key = "b" + str(i)
            a_prev_key = "A" + str(i - 1)

            W_curr = self.__weights[w_key]
            A_prev = cache[a_prev_key]

            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            if i > 1:
                dz = np.dot(W_curr.T, dz) * (A_prev * (1 - A_prev))

            self.__weights[w_key] -= alpha * dw
            self.__weights[b_key] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network.
        iterations: number of training iterations.
        alpha: learning rate.
        Returns the evaluation after training.
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
            self.gradient_descent(Y, self.cache, alpha)

        return self.evaluate(X, Y)
