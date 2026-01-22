#!/usr/bin/env python3
"""Calculate integral of a polynomial."""


def poly_integral(poly, C=0):
    """Calculate integral of polynomial given as coefficients list.

    Args:
        poly: List of coefficients [a0, a1, a2, ...]
              for a0 + a1*x + a2*x² + ...
        C: Integration constant

    Returns:
        List of coefficients of integral, or None if invalid input
    """
    # Check if poly is valid
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check if all coefficients are numbers
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Check if C is a number
    if not isinstance(C, (int, float)):
        return None

    # Integral: ∫ a_n*x^n dx = a_n/(n+1) * x^(n+1) + C
    integral = [C]  # Start with integration constant

    for power, coeff in enumerate(poly):
        # ∫ coeff * x^power dx = coeff/(power+1) * x^(power+1)
        new_coeff = coeff / (power + 1)

        # If coefficient is whole number, represent as integer
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)

        integral.append(new_coeff)

    # Remove trailing zeros to make list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
