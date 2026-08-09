#!/usr/bin/env python3
"""Module that contains the play function."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: the FrozenLakeEnv instance
        Q (numpy.ndarray): the Q-table
        max_steps (int): the maximum number of steps in the episode

    Returns:
        total_rewards, rendered_outputs
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = [env.render()]

    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, _ = env.step(action)

        rendered_outputs.append(env.render())

        state = new_state
        total_rewards += reward

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
