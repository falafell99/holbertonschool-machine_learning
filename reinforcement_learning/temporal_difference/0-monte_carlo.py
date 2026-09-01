#!/usr/bin/env python3
"""
Module for the Monte Carlo algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                 gamma=0.99):
    """
    Performs the Monte Carlo algorithm
    Args:
        env: The environment instance
        V: A numpy.ndarray of shape (s,) containing the value estimate
        policy: A function that takes in a state and returns the next
            action to take
        episodes: The total number of episodes to train over
        max_steps: The maximum number of steps per episode
        alpha: The learning rate
        gamma: The discount rate
    Returns:
        V, the updated value estimate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_states = []
        episode_rewards = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_states.append(state)
            episode_rewards.append(reward)
            state = next_state

            if terminated or truncated:
                break

        episode_states = np.array(episode_states)
        episode_rewards = np.array(episode_rewards)

        G = 0
        for t in reversed(range(len(episode_states))):
            state_t = episode_states[t]
            reward_t = episode_rewards[t]
            G = gamma * G + reward_t
            if state_t not in episode_states[:t]:
                V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
