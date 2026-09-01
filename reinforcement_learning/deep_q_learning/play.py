#!/usr/bin/env python3
"""
Module to display a game of Atari's Breakout played by a trained agent
"""
import numpy as np
import gymnasium as gym
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

from train import (
    GymnasiumCompatWrapper, AtariProcessor, build_model, IMG_SHAPE,
    WINDOW_LENGTH
)


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = GymnasiumCompatWrapper(env)
    np.random.seed(23)
    env.action_space.seed(23)
    nb_actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, IMG_SHAPE, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.
    )
    dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=5, visualize=True)
