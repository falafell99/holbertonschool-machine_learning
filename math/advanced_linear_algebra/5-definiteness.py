#!/usr/bin/env python3
"""Calculate definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """Calculate definiteness of a matrix.

    Args:
        matrix: numpy.ndarray of shape (n, n) whose definiteness should be
               calculated

    Returns:
        String: Positive definite, Positive semi-definite, Negative semi-definite,
               Negative definite, or Indefinite
        None: If matrix is not valid or doesn't fit any category

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues = np.round(eigenvalues, decimals=10)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
