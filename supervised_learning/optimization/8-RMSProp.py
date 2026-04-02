#!/usr/bin/env python3
"""Module to create an RMSProp optimizer in TensorFlow."""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.
    alpha: learning rate.
    beta2: RMSProp weight (discounting factor).
    epsilon: small number to avoid division by zero.
    Returns: the optimizer.
    """
    # В Keras RMSprop параметр beta2 соответствует аргументу rho
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
