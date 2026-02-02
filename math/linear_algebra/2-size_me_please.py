#!/usr/bin/env python3
"""Calculate shape of a matrix."""


def matrix_shape(matrix):
    """Calculate shape (dimensions) of a matrix.

    Args:
        matrix: Input matrix (can be nested lists)

    Returns:
        List of integers representing matrix dimensions
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if matrix:  # Check if list is not empty
            matrix = matrix[0]
        else:
            break
    return shape
