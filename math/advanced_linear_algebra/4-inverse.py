#!/usr/bin/env python3
"""Calculate inverse of a matrix."""


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


def cofactor(matrix):
    """Calculate cofactor matrix of a matrix.

    Args:
        matrix: List of lists whose cofactor matrix should be calculated

    Returns:
        The cofactor matrix of matrix

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
    if (matrix == [[]] or len(matrix) == 0 or
            any(len(row) == 0 for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Create cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Create submatrix without i-th row and j-th column
            submatrix = []
            for k in range(n):
                if k == i:
                    continue
                row = []
                for m in range(n):
                    if m == j:
                        continue
                    row.append(matrix[k][m])
                submatrix.append(row)

            # Calculate determinant of submatrix with sign (-1)^(i+j)
            sign = 1 if (i + j) % 2 == 0 else -1
            cofactor_value = sign * determinant(submatrix)
            cofactor_row.append(cofactor_value)

        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """Calculate adjugate matrix of a matrix.

    Args:
        matrix: List of lists whose adjugate matrix should be calculated

    Returns:
        The adjugate matrix of matrix

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
    if (matrix == [[]] or len(matrix) == 0 or
            any(len(row) == 0 for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Calculate cofactor matrix
    cofactor_mat = cofactor(matrix)

    # Transpose the cofactor matrix to get adjugate
    adjugate_matrix = []
    for j in range(n):
        adjugate_row = []
        for i in range(n):
            adjugate_row.append(cofactor_mat[i][j])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix


def inverse(matrix):
    """Calculate inverse of a matrix.

    Args:
        matrix: List of lists whose inverse should be calculated

    Returns:
        The inverse of matrix, or None if matrix is singular

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
    if (matrix == [[]] or len(matrix) == 0 or
            any(len(row) == 0 for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Calculate determinant
    det = determinant(matrix)

    # Check if matrix is singular
    if det == 0:
        return None

    # Special case: 1x1 matrix
    if n == 1:
        return [[1 / det]]

    # Calculate adjugate matrix
    adj = adjugate(matrix)

    # Calculate inverse: adjugate / determinant
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
