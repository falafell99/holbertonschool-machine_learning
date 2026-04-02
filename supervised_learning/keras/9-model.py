#!/usr/bin/env python3
"""Module for saving and loading Keras models."""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.
    network: the model to save.
    filename: the path to the file.
    Returns: None.
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model from a file.
    filename: the path to the file.
    Returns: the loaded model.
    """
    return K.models.load_model(filename)
