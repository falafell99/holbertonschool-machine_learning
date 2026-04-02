#!/usr/bin/env python3
"""Module to create a learning rate decay operation in TensorFlow."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TF using inverse time decay.
    alpha: the original learning rate.
    decay_rate: the weight used to determine the rate of decay.
    decay_step: the number of passes before alpha is decayed further.
    Returns: the learning rate decay operation (schedule).
    """
    # staircase=True обеспечивает ступенчатое (stepwise) затухание
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
