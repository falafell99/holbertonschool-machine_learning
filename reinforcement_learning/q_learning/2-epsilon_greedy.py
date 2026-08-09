#!/usr/bin/env python3
"""Module that contains the epsilon_greedy function."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q (numpy.ndarray): the q-table
        state: the current state
        epsilon (float): the epsilon to use for the calculation

    Returns:
        int: the next action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
