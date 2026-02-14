#!/usr/bin/env python3
"""Poisson distribution class."""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)) / len(data)

    def pmf(self, k):
        """Calculate PMF value for k successes."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = 2.718281828459045
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return (e ** -self.lambtha) * (self.lambtha ** k) / factorial

    def cdf(self, k):
        """Calculate CDF value for k successes."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = 2.718281828459045
        cdf_value = 0
        for i in range(k + 1):
            factorial = 1
            for j in range(1, i + 1):
                factorial *= j
            cdf_value += (self.lambtha ** i) / factorial

        return (e ** -self.lambtha) * cdf_value
