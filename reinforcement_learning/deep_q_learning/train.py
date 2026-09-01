#!/usr/bin/env python3
"""
Module to train an agent to play Atari's Breakout using DQN
"""
import numpy as np
import gymnasium as gym
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Permute, Activation
)
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint

IMG_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class GymnasiumCompatWrapper(gym.Wrapper):
    """
    Wrapper to make a gymnasium environment compatible with keras-rl2,
    which expects the older gym API (reset returns obs only, step
    returns a 4-tuple, render takes no mode argument at call time).
    """
    def reset(self, **kwargs):
        """Resets the environment, returning only the observation"""
        observation, info = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """
        Steps the environment, combining terminated and truncated into
        a single done flag
        """
        observation, reward, terminated, truncated, info = \
            self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        """Renders the environment using the mode set at creation"""
        return self.env.render()


class AtariProcessor(Processor):
    """
    Processor to preprocess Atari frames: grayscale, resize, and
    normalize for use with the DQN agent
    """
    def process_observation(self, observation):
        """Converts a raw frame into a preprocessed grayscale frame"""
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == IMG_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """Normalizes a batch of stacked frames to the range [0, 1]"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Clips the reward between -1 and 1"""
        return np.clip(reward, -1., 1.)


def build_model(window_length, shape, actions):
    """
    Builds the convolutional neural network used to approximate the
    Q-value function
    Args:
        window_length: The number of stacked frames per input
        shape: The (height, width) of a single frame
        actions: The number of possible actions
    Returns:
        The compiled keras model
    """
    input_shape = (window_length,) + shape

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('linear'))

    return model


def build_agent(model, actions):
    """
    Builds the DQN agent
    Args:
        model: The keras model approximating the Q-value function
        actions: The number of possible actions
    Returns:
        The compiled DQN agent
    """
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1.,
        value_min=.1, value_test=.05, nb_steps=1000000
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=actions,
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

    return dqn


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5')
    env = GymnasiumCompatWrapper(env)
    np.random.seed(23)
    env.action_space.seed(23)
    nb_actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, IMG_SHAPE, nb_actions)
    dqn = build_agent(model, nb_actions)

    checkpoint_callback = ModelIntervalCheckpoint(
        'policy_checkpoint.h5', interval=100000
    )

    dqn.fit(
        env,
        nb_steps=1750000,
        log_interval=10000,
        callbacks=[checkpoint_callback],
        visualize=False,
        verbose=2
    )

    dqn.save_weights('policy.h5', overwrite=True)
