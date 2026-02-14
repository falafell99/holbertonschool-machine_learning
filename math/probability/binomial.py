#!/usr/bin/env python3
"""Binomial distribution class."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize binomial distribution.

        Args:
            data: List of data to estimate the distribution
            n: Number of Bernoulli trials
            p: Probability of a success

        Raises:
            ValueError: If n is not positive or p is not valid probability
            TypeError: If data is not a list
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = float(sum(data) / len(data))
            variance = 0
            for x in data:
                variance += (x - mean) ** 2
            variance /= len(data)

            p = 1 - (variance / mean)
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculate PMF value for k successes.

        Args:
            k: Number of successes

        Returns:
            PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Calculate combination: n choose k
        comb = 1
        for i in range(1, k + 1):
            comb *= (self.n - k + i) / i

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

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
        if k >= self.n:
            return 1

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
