#!/usr/bin/env python3
"""Module that contains the load_frozen_lake function."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc (list): either None or a list of lists containing a
            custom description of the map to load for the
            environment
        map_name (str): either None or a string containing the
            pre-made map to load
        is_slippery (bool): determines if the ice is slippery

    Returns:
        The environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
    )
    return env
