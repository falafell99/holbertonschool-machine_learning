#!/usr/bin/env python3
"""Module to create a batch normalization layer in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.
    prev: the activated output of the previous layer.
    n: the number of nodes in the layer to be created.
    activation: the activation function to be used on the output.
    Returns: a tensor of the activated output for the layer.
    """
    # Инициализатор по условию
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # ВАЖНО: use_bias=False — это ключ к тому, чтобы получить 0.231
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )(prev)

    # BatchNormalization с нужным эпсилоном
    # Gamma и Beta по умолчанию инициализируются как 1 и 0
    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7
    )(dense)

    # Активация применяется в самом конце
    return activation(batch_norm)
