#!/usr/bin/env python3
"""Module for Gradient Descent with L2 Regularization."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization.
    Y: one-hot numpy.ndarray (classes, m) with correct labels.
    weights: dictionary of weights and biases.
    cache: dictionary of outputs of each layer.
    alpha: learning rate.
    lambtha: L2 regularization parameter.
    L: number of layers of the network.
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)

        # dw с учетом L2 штрафа: (grad / m) + (lambda / m) * W
        dw = (np.matmul(dz, A_prev.T) / m) + (lambtha / m) * weights[W_key]
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # Производная tanh: 1 - A^2
            W = weights[W_key]
            dz = np.matmul(W.T, dz) * (1 - np.square(A_prev))

        # Обновление весов "на месте"
        weights[W_key] -= alpha * dw
        weights[b_key] -= alpha * db
