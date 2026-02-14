#!/usr/bin/env python3
"""Module that calculates the likelihood of obtaining data"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array of hypothetical probabilities

    Returns:
        numpy.ndarray: 1D array containing the likelihood of obtaining the
        data, x and n, for each probability in P
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

    # Вычисление биномиального коэффициента (комбинации) nCr
    # Используем встроенный math.factorial и целочисленное деление,
    # чтобы избежать переполнения float при огромных факториалах
    import math
    fact_n = math.factorial(n)
    fact_x = math.factorial(x)
    fact_nx = math.factorial(n - x)
    combo = fact_n // (fact_x * fact_nx)

    # Вычисляем правдоподобие по биномиальной формуле
    likelihood_values = combo * (P ** x) * ((1 - P) ** (n - x))

    return likelihood_values
