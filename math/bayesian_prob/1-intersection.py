#!/usr/bin/env python3
"""Module to calculate intersection"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining data with various
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    def factorial(n):
        """Helper to calculate factorial"""
        res = 1
        for i in range(1, n + 1):
            res *= i
        return res

    # Вычисляем Likelihood (правдоподобие)
    num = factorial(n)
    den = factorial(x) * factorial(n - x)
    combination = float(num // den)
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = Likelihood * Prior
    return likelihood * Pr
