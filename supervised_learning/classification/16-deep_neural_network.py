#!/usr/bin/env python3
"""Module defining a deep neural network."""
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

            # Input size for the current layer
            if i == 0:
                prev_nodes = nx
            else:
                prev_nodes = layers[i - 1]

            # He et al. initialization: randn * sqrt(2 / prev_nodes)
            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            self.weights[w_key] = np.random.randn(layers[i], prev_nodes) * \
                np.sqrt(2 / prev_nodes)
            self.weights[b_key] = np.zeros((layers[i], 1))
