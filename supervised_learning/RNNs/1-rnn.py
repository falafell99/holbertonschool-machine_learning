#!/usr/bin/env python3
"""Module that contains the rnn function."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: An instance of RNNCell that will be used for the
            forward propagation.
        X: A numpy.ndarray of shape (t, m, i) that contains the data to be
            used.
            - t is the maximum number of time steps.
            - m is the batch size.
            - i is the dimensionality of the data.
        h_0: A numpy.ndarray of shape (m, h) containing the initial hidden
            state.
            - h is the dimensionality of the hidden state.

    Returns:
        tuple: (H, Y)
        - H is a numpy.ndarray containing all of the hidden states.
        - Y is a numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Мы можем достать размерность выхода 'o' из смещения (bias) ячейки
    o = rnn_cell.by.shape[1]

    # Инициализируем H (t + 1 шаг, так как мы храним h_0) и Y (t шагов)
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Записываем начальное скрытое состояние на позицию 0
    H[0] = h_0

    # Проходим по всем временным шагам (unroll the RNN)
    for step in range(t):
        # Передаем предыдущее состояние и текущий вход в ячейку
        h_next, y_t = rnn_cell.forward(H[step], X[step])

        # Сохраняем новые значения
        H[step + 1] = h_next
        Y[step] = y_t

    return H, Y
