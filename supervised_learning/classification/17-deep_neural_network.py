#!/usr/bin/env python3
"""Module defining a deep neural network with private attributes."""
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

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev_nodes = nx
            else:
                prev_nodes = layers[i - 1]

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            # He et al. initialization
            self.__weights[w_key] = np.random.randn(layers[i], prev_nodes) * \
                np.sqrt(2 / prev_nodes)
            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers __L."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary __cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary __weights."""
        return self.__weights
