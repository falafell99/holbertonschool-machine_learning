#!/usr/bin/env python3
"""Concatenate 2D matrices along specified axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along specified axis.

    Args:
        mat1: First 2D matrix
        mat2: Second 2D matrix
        axis: 0 for vertical (add rows), 1 for horizontal (add columns)

    Returns:
        New concatenated matrix, or None if shapes incompatible
    """
    # Check if matrices are valid
    if not mat1 or not mat2:
        return None

    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])

    if axis == 0:
        # Vertical concatenation: add rows, columns must match
        if cols1 != cols2:
            return None

        # Create deep copy and add rows from mat2
        result = [row.copy() for row in mat1]
        for row in mat2:
            result.append(row.copy())
        return result

    elif axis == 1:
        # Horizontal concatenation: add columns, rows must match
        if rows1 != rows2:
            return None

        # Create deep copy and extend each row with mat2 columns
        result = []
        for i in range(rows1):
            new_row = mat1[i].copy()
            new_row.extend(mat2[i].copy())
            result.append(new_row)
        return result

    else:
        # Invalid axis
        return None
