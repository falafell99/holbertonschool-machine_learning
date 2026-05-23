#!/usr/bin/env python3
"""Module to perform K-means clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    # 1st use of np.random.uniform: Initialize all centroids
    C = np.random.uniform(low=low, high=high, size=(k, d))

    for i in range(iterations):
        C_prev = np.copy(C)

        # Assignment Step: Calculate distances and assign clusters
        # X shape: (n, 1, d), C shape: (1, k, d) -> Broadcasting to (n, k, d)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        # Update Step: Recalculate centroids
        for j in range(k):
            cluster_points = X[clss == j]
            if len(cluster_points) == 0:
                # 2nd use of np.random.uniform: Reinitialize empty cluster
                C[j] = np.random.uniform(low=low, high=high, size=(1, d))
            else:
                C[j] = np.mean(cluster_points, axis=0)

        # Convergence Check: Stop early if centroids haven't moved
        if np.array_equal(C_prev, C):
            break

    # Final assignment to ensure `clss` matches the final updated `C`
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
    clss = np.argmin(distances, axis=1)

    return C, clss
