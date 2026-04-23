#!/usr/bin/env python3
"""Module to create an identity block for ResNet."""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015).

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3, F12:
            F11: number of filters in the first 1x1 convolution.
            F3: number of filters in the 3x3 convolution.
            F12: number of filters in the second 1x1 convolution.

    Returns:
        The activated output of the identity block.
    """
    F11, F3, F12 = filters
    
    # Инициализатор весов He Normal с фиксированным seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # --- Первый компонент основного пути ---
    # 1x1 свертка для уменьшения размерности (bottleneck)
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # --- Второй компонент основного пути ---
    # 3x3 свертка для извлечения признаков
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # --- Третий компонент основного пути ---
    # 1x1 свертка для восстановления размерности каналов
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # --- Сложение и финальная активация ---
    # Добавляем "Shortcut" (исходный вход A_prev) к результату сверток
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
