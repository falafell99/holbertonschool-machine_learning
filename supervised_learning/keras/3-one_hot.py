#!/usr/bin/env python3
"""Module to convert labels to one-hot matrix using Keras."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    labels: the labels to convert.
    classes: the total number of classes.
    Returns: the one-hot matrix.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
