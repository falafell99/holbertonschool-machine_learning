#!/usr/bin/env python3
"""Calculate minor matrix of a matrix."""


def determinant(matrix):
    """Calculate determinant of a square matrix.

    Args:
        matrix: List of lists representing the matrix

    Returns:
        Determinant value

    Raises:
        TypeError: If matrix is not list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Special case: 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        a, b = matrix[0][0], matrix[0][1]
        c, d = matrix[1][0], matrix[1][1]
        return a * d - b * c

    # Recursive case: Laplace expansion along first row
    det = 0
    for j in range(n):
        # Create submatrix without first row and j-th column
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)

        # Calculate minor with sign (-1)^j
        sign = 1 if j % 2 == 0 else -1
        det += sign * matrix[0][j] * determinant(submatrix)

    return det


def minor(matrix):
    """Calculate minor matrix of a matrix.

    Args:
        matrix: List of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check for empty list
    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    # Check if all elements are lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty (contains empty list or empty rows)
    if matrix == [[]] or len(matrix) == 0 or any(len(row) == 0 for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Create minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Create submatrix without i-th row and j-th column
            submatrix = []
            for k in range(n):
                if k == i:
                    continue
                row = []
                for l in range(n):
                    if l == j:
                        continue
                    row.append(matrix[k][l])
                submatrix.append(row)

            # Calculate determinant of submatrix
            minor_value = determinant(submatrix)
            minor_row.append(minor_value)

        minor_matrix.append(minor_row)

    return minor_matrix
