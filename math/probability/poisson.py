#!/usr/bin/env python3
"""Poisson distribution class."""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution.

        Args:
            data: List of data to estimate the distribution
            lambtha: Expected number of occurrences in a given time frame

        Raises:
            ValueError: If lambtha is not positive or data has less than 2
                       points
            TypeError: If data is not a list
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate PMF value for k successes.

        Args:
            k: Number of successes

        Returns:
            PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        e = 2.718281828459045
        return (e ** -self.lambtha) * (self.lambtha ** k) / factorial

    def cdf(self, k):
        """Calculate CDF value for k successes.

        Args:
            k: Number of successes

        Returns:
            CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            fact = 1
            for j in range(1, i + 1):
                fact *= j
            e = 2.718281828459045
            cdf_value += (e ** -self.lambtha) * (self.lambtha ** i) / fact

        return cdf_value
