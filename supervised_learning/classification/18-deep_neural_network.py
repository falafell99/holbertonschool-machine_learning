#!/usr/bin/env python3
"""Module defining a deep neural network with forward propagation."""
import numpy as np


class DeepNeuralNetwork:
    """Class representing a deep neural network."""

    def __init__(self, nx, layers):
        """
        Initializes deep neural network.
        nx: number of input features.
        layers: list representing nodes in each layer.
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

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if i == 0 else layers[i - 1]

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            # He et al. initialization
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
        """
        Calculates forward propagation for the deep neural network.
        X: input data of shape (nx, m).
        Updates and returns the output and cache.
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            a_prev_key = "A" + str(i)
            a_curr_key = "A" + str(i + 1)

            # Z = W * A_prev + b
            Z = np.dot(self.__weights[w_key], self.__cache[a_prev_key]) + \
                self.__weights[b_key]

            # Sigmoid activation
            self.__cache[a_curr_key] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache
