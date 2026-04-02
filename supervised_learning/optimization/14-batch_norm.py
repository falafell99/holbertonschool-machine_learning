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
    # 1. Инициализатор весов VarianceScaling с модом fan_avg
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Создаем базовый Dense слой.
    # use_bias=False, так как параметр beta в батч-норме заменяет bias.
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )(prev)

    # 3. Добавляем слой Batch Normalization.
    # gamma и beta обучаются автоматически, epsilon задан по условию.
    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer='ones',
        beta_initializer='zeros',
        epsilon=1e-7
    )(dense)

    # 4. Применяем функцию активации к нормализованному выходу
    return activation(batch_norm)
