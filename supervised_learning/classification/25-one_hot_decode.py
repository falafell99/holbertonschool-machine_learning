#!/usr/bin/env python3
"""Module that converts a one-hot matrix to a label vector."""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.
    one_hot: numpy.ndarray of shape (classes, m).
    Returns: numpy.ndarray of shape (m,) or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        # argmax finds the index of the maximum value along the specified axis.
        # axis=0 means we look through the rows (classes) for each column.
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
