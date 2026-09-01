#!/usr/bin/env python3
"""
Module for the SARSA(λ) algorithm
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Chooses an action using the epsilon greedy policy
    Args:
        Q: A numpy.ndarray containing the Q table
        state: The current state
        epsilon: The epsilon value to use for the calculation
    Returns:
        The next action index
    """
    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                   gamma=0.99, epsilon=1, min_epsilon=0.1,
                   epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm
    Args:
        env: The environment instance
        Q: A numpy.ndarray of shape (s,a) containing the Q table
        lambtha: The eligibility trace factor
        episodes: The total number of episodes to train over
        max_steps: The maximum number of steps per episode
        alpha: The learning rate
        gamma: The discount rate
        epsilon: The initial threshold for epsilon greedy
        min_epsilon: The minimum value that epsilon should decay to
        epsilon_decay: The decay rate for updating epsilon between
            episodes
    Returns:
        Q, the updated Q table
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility_trace = np.zeros(Q.shape)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state, next_action] - \
                Q[state, action]
            eligibility_trace[state, action] += 1

            Q = Q + alpha * delta * eligibility_trace
            eligibility_trace = gamma * lambtha * eligibility_trace

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

    return Q
