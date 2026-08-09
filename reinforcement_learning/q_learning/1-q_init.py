#!/usr/bin/env python3
"""Module that contains the q_init function."""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        numpy.ndarray: the Q-table, initialized as zeros
    """
    state_space = env.observation_space.n
    action_space = env.action_space.n
    return np.zeros((state_space, action_space))
