#!/usr/bin/env python3
"""
Module for training an agent with the Monte-Carlo policy gradient
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training
    Args:
        env: The initial environment
        nb_episodes: The number of episodes used for training
        alpha: The learning rate
        gamma: The discount factor
    Returns:
        All values of the score (sum of all rewards during one episode
        loop)
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        done = False

        while not done:
            action, gradient = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            gradients.append(gradient)
            rewards.append(reward)
            done = terminated or truncated

        for t in range(len(gradients)):
            discounts = gamma ** np.arange(len(rewards) - t)
            G = np.sum(np.array(rewards[t:]) * discounts)
            weight += alpha * gradients[t] * G

        score = sum(rewards)
        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
