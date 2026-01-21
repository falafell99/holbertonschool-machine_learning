#!/usr/bin/env python3
"""Calculate sum of squares from 1 to n."""


def summation_i_squared(n):
    """Calculate Σ from i=1 to n of i².
    
    Args:
        n: Stopping condition
        
    Returns:
        Integer sum of squares, or None if n invalid
    """
    if type(n) is not int or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
