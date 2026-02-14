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
