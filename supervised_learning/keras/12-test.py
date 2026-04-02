#!/usr/bin/env python3
"""Module to test a Keras model."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.
    network: the network model to test.
    data: input data to test the model with.
    labels: correct one-hot labels of data.
    verbose: boolean that determines if output should be printed.
    Returns: loss and accuracy of the model, respectively.
    """
    # evaluate returns a list: [loss, accuracy, ...]
    return network.evaluate(x=data, y=labels, verbose=verbose)
