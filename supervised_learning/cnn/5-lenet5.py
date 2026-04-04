#!/usr/bin/env python3
"""Module to build a modified LeNet-5 architecture using Keras."""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using keras.
    X: K.Input of shape (m, 28, 28, 1) containing the input images.
    Returns: a compiled K.Model.
    """
    init = K.initializers.HeNormal(seed=0)

    # C1: Convolutional layer (6 kernels 5x5, same padding, ReLU)
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # S2: Max pooling layer (2x2 kernels, 2x2 strides)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # C3: Convolutional layer (16 kernels 5x5, valid padding, ReLU)
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    # S4: Max pooling layer (2x2 kernels, 2x2 strides)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the tensor for fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # F5: Fully connected layer (120 nodes, ReLU)
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flatten)

    # F6: Fully connected layer (84 nodes, ReLU)
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    # F7: Output softmax layer (10 nodes)
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(fc2)

    model = K.Model(inputs=X, outputs=output)

    # Компиляция с Adam и метрикой accuracy
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
