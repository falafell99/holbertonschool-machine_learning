#!/usr/bin/env python3
"""Transpose a 2D matrix."""


def matrix_transpose(matrix):
    """Return transpose of a 2D matrix.

    Args:
        matrix: Input 2D matrix

    Returns:
        Transposed matrix
    """
    # Get dimensions
    rows = len(matrix)
    cols = len(matrix[0])

    # Create empty transposed matrix
    transpose = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(matrix[i][j])
        transpose.append(new_row)

    return transpose
