#!/usr/bin/env python3
"""Module to build a neural network with Keras Functional API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.
    nx: number of input features.
    layers: list containing the number of nodes in each layer.
    activations: list of activation functions for each layer.
    lambtha: L2 regularization parameter.
    keep_prob: probability that a node will be kept for dropout.
    Returns: the keras model.
    """
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.L2(lambtha)

    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=L2)(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
