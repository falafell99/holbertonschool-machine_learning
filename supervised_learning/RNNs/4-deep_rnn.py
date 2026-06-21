#!/usr/bin/env python3
"""Module that contains the deep_rnn function."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: A list of RNNCell instances of length l that will be
            used for the forward propagation.
        X: A numpy.ndarray of shape (t, m, i) that contains the data.
            - t is the maximum number of time steps.
            - m is the batch size.
            - i is the dimensionality of the data.
        h_0: A numpy.ndarray of shape (l, m, h) containing the initial
            hidden state.
            - l is the number of layers.
            - h is the dimensionality of the hidden state.

    Returns:
        tuple: (H, Y)
        - H is a numpy.ndarray containing all of the hidden states.
        - Y is a numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    l, m, h = h_0.shape

    # Размерность выхода берем из bias последней ячейки
    o = rnn_cells[-1].by.shape[1]

    # Инициализируем H и Y
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    # Проходим по каждому временному шагу
    for step in range(t):
        # На каждом временном шаге проходим по всем слоям снизу вверх
        for layer in range(l):
            # Вход для нулевого слоя - это данные X
            if layer == 0:
                x_t = X[step]
            # Вход для остальных слоев - выход предыдущего слоя на этом шаге
            else:
                x_t = H[step + 1, layer - 1]

            # Достаем скрытое состояние предыдущего шага для текущего слоя
            h_prev = H[step, layer]

            # Делаем шаг forward
            h_next, y_t = rnn_cells[layer].forward(h_prev, x_t)

            # Сохраняем новое скрытое состояние
            H[step + 1, layer] = h_next

            # Если это последний слой, сохраняем выход в Y
            if layer == l - 1:
                Y[step] = y_t

    return H, Y
