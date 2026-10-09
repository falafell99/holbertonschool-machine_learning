#!/usr/bin/env python3
"""
Module for the policy gradient policy function and Monte-Carlo gradient
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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a
    weight matrix
    Args:
        state: A matrix representing the current observation of the
            environment
        weight: A matrix of random weight
    Returns:
        The action and the gradient (in this order)
    """
    state = np.atleast_2d(state)
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    dsoftmax = -probs.copy()
    dsoftmax[0, action] += 1
    gradient = state.T.dot(dsoftmax)

    return action, gradient
