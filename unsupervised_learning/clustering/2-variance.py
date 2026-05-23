#!/usr/bin/env python3
"""Module to calculate intra-cluster variance"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    try:
        # Broadcasting: X becomes (n, 1, d) and C acts as (1, k, d)
        # diff shape: (n, k, d)
        diff = X[:, np.newaxis] - C

        # Calculate squared distances
        dist_sq = np.sum(np.square(diff), axis=2)

        # Find the minimum squared distance for each point (distance to its cluster)
        min_dist_sq = np.min(dist_sq, axis=1)

        # Total variance is the sum of these minimum squared distances
        var = np.sum(min_dist_sq)

        return float(var)
    except Exception:
        return None
