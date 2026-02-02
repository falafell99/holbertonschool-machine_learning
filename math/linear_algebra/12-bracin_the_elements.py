#!/usr/bin/env python3
"""Element-wise operations on numpy arrays."""


def np_elementwise(mat1, mat2):
    """Perform element-wise addition, subtraction,
    multiplication and division.

    Args:
        mat1: First numpy array or scalar
        mat2: Second numpy array or scalar

    Returns:
        Tuple of (sum, difference, product, quotient)
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
