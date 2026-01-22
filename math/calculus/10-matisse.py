#!/usr/bin/env python3
"""Calculate derivative of a polynomial."""


def poly_derivative(poly):
    """Calculate derivative of polynomial given as coefficients list.

    Args:
        poly: List of coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x² + ...

    Returns:
        List of coefficients of derivative, or None if invalid input
    """
    # Check if poly is valid
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check if all coefficients are numbers
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    # If polynomial is constant (degree 0)
    if len(poly) == 1:
        return [0]

    # Calculate derivative: derivative of a_n*x^n = n*a_n*x^(n-1)
    derivative = []
    for power in range(1, len(poly)):
        derivative.append(power * poly[power])

    # If derivative is all zeros (e.g., poly = [0, 0, 0])
    if all(c == 0 for c in derivative):
        return [0]

    return derivative
