#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""
import numpy as np


class Neuron:
    """Class representing a single neuron."""

    def __init__(self, nx):
        """
        Initializes the neuron.
        nx is the number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
