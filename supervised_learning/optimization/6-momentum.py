#!/usr/bin/env python3
"""Module to create a Momentum optimizer in TensorFlow."""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in TF.
    alpha: learning rate.
    beta1: momentum weight.
    Returns: the optimizer.
    """
    # В Keras/TensorFlow моментум — это расширение SGD
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
