#!/usr/bin/env python3
"""Module to calculate intra-cluster variance"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance of a data set.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    try:
        diff = X[:, np.newaxis] - C
        dist_sq = np.sum(np.square(diff), axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        var = np.sum(min_dist_sq)
        return float(var)
    except Exception:
        return None
