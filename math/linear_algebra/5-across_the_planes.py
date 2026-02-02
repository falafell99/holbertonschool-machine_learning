#!/usr/bin/env python3
"""Add two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Add two 2D matrices element by element.

    Args:
        mat1: First 2D matrix
        mat2: Second 2D matrix

    Returns:
        New matrix with element-wise sum, or None if shapes differ
    """
    # Check if matrices have same number of rows
    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        # Check if rows have same length
        if len(mat1[i]) != len(mat2[i]):
            return None

        # Add rows element-wise
        new_row = []
        for j in range(len(mat1[i])):
            new_row.append(mat1[i][j] + mat2[i][j])
        result.append(new_row)

    return result
