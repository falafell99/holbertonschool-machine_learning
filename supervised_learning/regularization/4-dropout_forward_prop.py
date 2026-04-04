#!/usr/bin/env python3
"""Module for Forward Propagation with Dropout."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    X: numpy.ndarray (nx, m) containing the input data.
    weights: dictionary of the weights and biases.
    L: number of layers in the network.
    keep_prob: probability that a node will be kept.
    Returns: dictionary containing outputs and dropout masks.
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        # Linear pass
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Last layer: Softmax
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Hidden layers: Tanh + Dropout
            A = np.tanh(Z)
            # Create mask (1 if keep, 0 if drop)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            # Inverted Dropout: apply mask and scale
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A

    return cache
