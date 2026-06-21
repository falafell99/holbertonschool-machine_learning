#!/usr/bin/env python3
"""Module that contains the LSTMCell class."""
import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory (LSTM) unit."""

    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Weights and biases for the forget gate
        self.Wf = np.random.normal(size=(h + i, h))
        self.bf = np.zeros((1, h))

        # Weights and biases for the update gate
        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros((1, h))

        # Weights and biases for the intermediate cell state
        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros((1, h))

        # Weights and biases for the output gate
        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros((1, h))

        # Weights and biases for the outputs
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): shape (m, h) containing the previous
                hidden state.
            c_prev (numpy.ndarray): shape (m, h) containing the previous
                cell state.
            x_t (numpy.ndarray): shape (m, i) that contains the data input
                for the cell.
                - m is the batch size for the data.

        Returns:
            tuple: (h_next, c_next, y)
            - h_next is the next hidden state.
            - c_next is the next cell state.
            - y is the output of the cell.
        """
        def sigmoid(x):
            """Helper function for the sigmoid activation."""
            return 1 / (1 + np.exp(-x))

        # Concatenate previous hidden state and input data
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # 1. Forget gate
        f = sigmoid(np.matmul(h_x, self.Wf) + self.bf)

        # 2. Update (Input) gate
        u = sigmoid(np.matmul(h_x, self.Wu) + self.bu)

        # 3. Intermediate cell state (candidate)
        c_tilde = np.tanh(np.matmul(h_x, self.Wc) + self.bc)

        # 4. Next cell state
        c_next = f * c_prev + u * c_tilde

        # 5. Output gate
        o = sigmoid(np.matmul(h_x, self.Wo) + self.bo)

        # 6. Next hidden state
        h_next = o * np.tanh(c_next)

        # 7. Output with softmax activation
        z_out = np.matmul(h_next, self.Wy) + self.by
        # Subtract max for numerical stability before exponentiation
        exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, c_next, y
