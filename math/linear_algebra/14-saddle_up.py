#!/usr/bin/env python3
"""Matrix multiplication using numpy."""

import numpy as np


def np_matmul(mat1, mat2):
    """Multiply two matrices using numpy.

    Args:
        mat1: First numpy array
        mat2: Second numpy array

    Returns:
        Result of matrix multiplication
    """
    return np.matmul(mat1, mat2)
