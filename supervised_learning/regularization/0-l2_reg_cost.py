#!/usr/bin/env python3
"""Module to calculate L2 regularization cost."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    cost: the cost of the network without L2 regularization.
    lambtha: the regularization parameter.
    weights: dictionary of the weights and biases (numpy.ndarrays).
    L: the number of layers in the neural network.
    m: the number of data points used.
    Returns: the cost of the network accounting for L2 regularization.
    """
    l2_term = 0
    for i in range(1, L + 1):
        key = 'W' + str(i)
        l2_term += np.sum(np.square(weights[key]))

    l2_cost = cost + (lambtha / (2 * m)) * l2_term
    return l2_cost
