#!/usr/bin/env python3
"""Module to build the ResNet-50 architecture."""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).

    Returns:
        The Keras model.
    """
    initializer = K.initializers.HeNormal(seed=0)
    input_tensor = K.Input(shape=(224, 224, 3))

    # --- STAGE 1 ---
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(input_tensor)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # --- STAGE 2 ---
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # --- STAGE 3 ---
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # --- STAGE 4 ---
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # --- STAGE 5 ---
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # --- POOLING & OUTPUT ---
    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        padding='same'
    )(X)
    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    model = K.models.Model(inputs=input_tensor, outputs=X)

    return model
