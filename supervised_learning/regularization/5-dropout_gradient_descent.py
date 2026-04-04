#!/usr/bin/env python3
"""Module for Gradient Descent with Dropout."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights of a NN with Dropout using gradient descent.
    Y: one-hot numpy.ndarray (classes, m) containing labels.
    weights: dictionary of weights and biases.
    cache: dictionary of outputs and dropout masks of each layer.
    alpha: learning rate.
    keep_prob: probability that a node will be kept.
    L: number of layers of the network.
    """
    m = Y.shape[1]
    # Градиент для последнего слоя (Softmax)
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)
        W = weights[W_key]

        # Вычисляем градиенты весов и смещений
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # Обратный проход через активацию tanh и маску Dropout
            da = np.matmul(W.T, dz)
            # Применяем маску из кеша предыдущего слоя
            da *= cache['D' + str(i - 1)]
            # Масштабируем градиент (Inverted Dropout)
            da /= keep_prob
            # Производная tanh: (1 - A^2)
            dz = da * (1 - np.square(A_prev))

        # Обновляем параметры in place
        weights[W_key] -= alpha * dw
        weights[b_key] -= alpha * db
