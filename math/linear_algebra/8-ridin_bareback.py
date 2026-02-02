#!/usr/bin/env python3
"""Multiply two matrices."""


def mat_mul(mat1, mat2):
    """Multiply two matrices.

    Args:
        mat1: First matrix (m × n)
        mat2: Second matrix (n × p)

    Returns:
        Result matrix (m × p), or None if dimensions incompatible
    """
    # Check if matrices can be multiplied
    # mat1 columns must equal mat2 rows
    if len(mat1[0]) != len(mat2):
        return None

    # Get dimensions
    m = len(mat1)      # rows of mat1
    n = len(mat1[0])   # columns of mat1 = rows of mat2
    p = len(mat2[0])   # columns of mat2

    # Initialize result matrix with zeros
    result = []
    for i in range(m):
        result.append([0] * p)

    # Perform matrix multiplication
    for i in range(m):           # for each row of mat1
        for j in range(p):       # for each column of mat2
            sum_val = 0
            for k in range(n):   # dot product
                sum_val += mat1[i][k] * mat2[k][j]
            result[i][j] = sum_val

    return result
