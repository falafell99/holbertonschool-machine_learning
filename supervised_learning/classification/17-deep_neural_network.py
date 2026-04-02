#!/usr/bin/env python3
"""Module defining a deep neural network with forward propagation."""
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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev_nodes = nx
            else:
                prev_nodes = layers[i - 1]

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            self.weights[w_key] = np.random.randn(layers[i], prev_nodes) * \
                np.sqrt(2 / prev_nodes)
            self.weights[b_key] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """
        Calculates forward propagation for the deep neural network.
        X: input data of shape (nx, m).
        Updates and returns the output and cache.
        """
        # A0 is the input data
        self.cache["A0"] = X

        for i in range(self.L):
            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            a_prev_key = "A" + str(i)
            a_curr_key = "A" + str(i + 1)

            # Z = W * A_prev + b
            Z = np.dot(self.weights[w_key], self.cache[a_prev_key]) + \
                self.weights[b_key]
            
            # Sigmoid activation: A = 1 / (1 + exp(-Z))
            self.cache[a_curr_key] = 1 / (1 + np.exp(-Z))

        return self.cache["A" + str(self.L)], self.cache
