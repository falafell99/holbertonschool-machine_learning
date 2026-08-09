#!/usr/bin/env python3
"""Module that contains the train function."""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning.

    Args:
        env: the FrozenLakeEnv instance
        Q (numpy.ndarray): the Q-table
        episodes (int): the total number of episodes to train over
        max_steps (int): the maximum number of steps per episode
        alpha (float): the learning rate
        gamma (float): the discount rate
        epsilon (float): the initial threshold for epsilon greedy
        min_epsilon (float): the minimum value epsilon should decay
            to
        epsilon_decay (float): the decay rate for updating epsilon
            between episodes

    Returns:
        Q, total_rewards
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            episode_reward += reward

            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

        total_rewards.append(episode_reward)

    return Q, total_rewards
