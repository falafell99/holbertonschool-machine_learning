#!/usr/bin/env python3
"""Module to calculate posterior probability"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data
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

    # 1. Likelihood
    fact_n = factorial(n)
    fact_x = factorial(x)
    fact_nx = factorial(n - x)
    combination = float(fact_n // (fact_x * fact_nx))
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    # 2. Intersection
    intersection = likelihood * Pr

    # 3. Marginal
    marginal_prob = np.sum(intersection)

    # 4. Posterior
    return intersection / marginal_prob
