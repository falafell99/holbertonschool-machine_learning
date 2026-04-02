#!/usr/bin/env python3
"""Module that converts numeric labels to one-hot matrix."""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.
    Y: numpy.ndarray of shape (m,) with numeric labels.
    classes: the maximum number of classes.
    Returns: one-hot matrix of shape (classes, m) or None on failure.
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int):
        return None

    try:
        # Create an identity matrix of size (classes x classes)
        # and index it using Y to get the one-hot rows, then transpose.
        one_hot = np.eye(classes)[Y].T
        return one_hot
    except Exception:
        # This handles cases where labels in Y >= classes or negative labels
        return None
