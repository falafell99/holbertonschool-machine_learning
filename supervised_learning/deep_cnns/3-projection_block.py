#!/usr/bin/env python3
"""Module to create a projection block for ResNet."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015).

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3, F12:
            F11: number of filters in the first 1x1 convolution.
            F3: number of filters in the 3x3 convolution.
            F12: number of filters in the second 1x1 convolution.
        s: stride of the first convolution in both paths.

    Returns:
        The activated output of the projection block.
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    # --- MAIN PATH ---

    # First component: 1x1 Convolution with stride 's'
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component: 3x3 Convolution
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component: 1x1 Convolution
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # --- SHORTCUT PATH ---

    # 1x1 Convolution to match dimensions with the main path
    X_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # --- Final Addition ---
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
