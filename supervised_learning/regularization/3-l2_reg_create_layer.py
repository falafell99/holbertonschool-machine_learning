#!/usr/bin/env python3
"""Module to create a Keras layer with L2 regularization."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow with L2 regularization.
    prev: tensor containing the output of the previous layer.
    n: number of nodes the new layer should contain.
    activation: the activation function to be used on the layer.
    lambtha: the L2 regularization parameter.
    Returns: the output of the new layer.
    """
    # Инициализатор весов (традиционно fan_avg для нейронок в ELTE/Holberton)
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Создаем слой с регуляризатором ядра (весов)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha),
        kernel_initializer=init
    )

    return layer(prev)
