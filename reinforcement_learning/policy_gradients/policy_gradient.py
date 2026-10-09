#!/usr/bin/env python3
"""
Module for the policy gradient policy function
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix
    Args:
        matrix: A numpy.ndarray of shape (n, s) containing the states
        weight: A numpy.ndarray of shape (s, a) containing the weights
    Returns:
        A numpy.ndarray of shape (n, a) containing the probability of
        each action (softmax of the matrix product)
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)
