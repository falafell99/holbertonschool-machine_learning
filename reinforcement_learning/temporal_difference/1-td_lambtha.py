#!/usr/bin/env python3
"""
Module for the TD(λ) algorithm
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm
    Args:
        env: The environment instance
        V: A numpy.ndarray of shape (s,) containing the value estimate
        policy: A function that takes in a state and returns the next
            action to take
        lambtha: The eligibility trace factor
        episodes: The total number of episodes to train over
        max_steps: The maximum number of steps per episode
        alpha: The learning rate
        gamma: The discount rate
    Returns:
        V, the updated value estimate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        eligibility_trace = np.zeros(V.shape[0])

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility_trace[state] += 1

            V = V + alpha * delta * eligibility_trace
            eligibility_trace = gamma * lambtha * eligibility_trace

            state = next_state

            if terminated or truncated:
                break

    return V
