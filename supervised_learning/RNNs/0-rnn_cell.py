#!/usr/bin/env python3
"""Module that contains the RNNCell class."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Initializes the RNNCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Weights for concatenated hidden state and input data
        self.Wh = np.random.normal(size=(h + i, h))
        # Weights for the output
        self.Wy = np.random.normal(size=(h, o))
        # Biases initialized to zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): shape (m, h) containing the previous
                hidden state.
            x_t (numpy.ndarray): shape (m, i) that contains the data input
                for the cell.
                - m is the batch size for the data.

        Returns:
            tuple: (h_next, y)
            - h_next is the next hidden state.
            - y is the output of the cell.
        """
        # Concatenate previous hidden state and input data
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state with tanh activation function
        h_next = np.tanh(np.matmul(h_x, self.Wh) + self.bh)

        # Compute the output logits
        z = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation function (numerically stable)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y
