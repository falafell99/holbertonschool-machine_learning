#!/usr/bin/env python3
"""Module to calculate likelihood"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various
    hypothetical probabilities
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    def factorial(n):
        """Helper to calculate factorial"""
        res = 1
        for i in range(1, n + 1):
            res *= i
        return res

    # Binomial coefficient: n! / (x! * (n - x)!)
    # Force to float to avoid numpy object array conversion with huge ints
    num = factorial(n)
    den = factorial(x) * factorial(n - x)
    combination = float(num // den)

    # Likelihood formula: nCr * P^x * (1-P)^(n-x)
    l_values = combination * (P ** x) * ((1 - P) ** (n - x))

    return l_values
