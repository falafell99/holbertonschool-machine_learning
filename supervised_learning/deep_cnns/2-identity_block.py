#!/usr/bin/env python3
"""Module to create an identity block for ResNet."""
from tensorflow import keras as K


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

    # Initialize weights using He Normal with seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # --- First component of main path ---
    # 1x1 convolution to reduce dimensionality (bottleneck)
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # --- Second component of main path ---
    # 3x3 convolution to extract features
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # --- Third component of main path ---
    # 1x1 convolution to restore channel dimensions
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # --- Addition and final activation ---
    # Shortcut connection: adding the original input A_prev to the main path
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
