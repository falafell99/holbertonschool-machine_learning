#!/usr/bin/env python3
"""Concatenate numpy arrays along axis."""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate two numpy arrays along specified axis.

    Args:
        mat1: First numpy array
        mat2: Second numpy array
        axis: 0 for vertical, 1 for horizontal concatenation

    Returns:
        Concatenated numpy array
    """
    return np.concatenate((mat1, mat2), axis=axis)
