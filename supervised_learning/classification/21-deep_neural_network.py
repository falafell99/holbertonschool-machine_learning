#!/usr/bin/env python3
"""Module defining a deep neural network with gradient descent."""
import numpy as np


class DeepNeuralNetwork:
    """Class representing a deep neural network."""

    def __init__(self, nx, layers):
        """
        Initializes deep neural network.
        nx: number of input features.
        layers: list of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Цикл 1: Инициализация весов и проверка элементов слоев
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if i == 0 else layers[i - 1]
            w_key, b_key = "W" + str(i + 1), "b" + str(i + 1)

            self.__weights[w_key] = np.random.randn(layers[i], prev_nodes) * \
                np.sqrt(2 / prev_nodes)
            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter __L."""
        return self.__L

    @property
    def cache(self):
        """Getter __cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter __weights."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation.
        X: input data.
        """
        self.__cache["A0"] = X
        # Цикл 2: Прямое распространение по слоям
        for i in range(self.__L):
            w_key, b_key = "W" + str(i + 1), "b" + str(i + 1)
            a_prev, a_curr = "A" + str(i), "A" + str(i + 1)

            Z = np.dot(self.__weights[w_key], self.__cache[a_prev]) + \
                self.__weights[b_key]
            self.__cache[a_curr] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates cost."""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent.
        Y: labels.
        cache: intermediary values.
        alpha: learning rate.
        """
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y

        # Цикл 3: Обратное распространение (от L до 1)
        for i in range(self.__L, 0, -1):
            w_key, b_key = "W" + str(i), "b" + str(i)
            a_prev_key = "A" + str(i - 1)

            W_curr = self.__weights[w_key]
            A_prev = cache[a_prev_key]

            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            if i > 1:
                dz = np.dot(W_curr.T, dz) * (A_prev * (1 - A_prev))

            self.__weights[w_key] -= alpha * dw
            self.__weights[b_key] -= alpha * db
