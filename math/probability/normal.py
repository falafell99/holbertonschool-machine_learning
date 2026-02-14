#!/usr/bin/env python3
"""Normal distribution class."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize normal distribution.

        Args:
            data: List of data to estimate the distribution
            mean: Mean of the distribution
            stddev: Standard deviation of the distribution

        Raises:
            ValueError: If stddev is not positive or data has less than 2
                       points
            TypeError: If data is not a list
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = 0
            for x in data:
                variance += (x - self.mean) ** 2
            variance /= len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculate z-score of x.

        Args:
            x: x-value

        Returns:
            z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate x-value of z-score.

        Args:
            z: z-score

        Returns:
            x-value of z
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """Calculate PDF value for x.

        Args:
            x: x-value

        Returns:
            PDF value for x
        """
        pi = 3.141592653589793
        e = 2.718281828459045
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))
        return coefficient * (e ** exponent)

    def cdf(self, x):
        """Calculate CDF value for x.

        Args:
            x: x-value

        Returns:
            CDF value for x
        """
        pi = 3.141592653589793
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (pi ** 0.5)) * (z - (z ** 3) / 3 + (z ** 5) / 10 -
                                    (z ** 7) / 42 + (z ** 9) / 216)
        return (1 + erf) / 2
