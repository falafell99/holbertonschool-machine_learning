#!/usr/bin/env python3
"""Module to build a neural network with Keras."""
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
    model = K.Sequential()
    L2 = K.regularizers.L2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # First layer needs input_shape
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=L2,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=L2))

        # Add dropout after every layer except the last one
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
