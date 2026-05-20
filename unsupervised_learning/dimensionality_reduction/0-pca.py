#!/usr/bin/env python3
"""Module for Principal Component Analysis"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    U, S, Vh = np.linalg.svd(X)
    cumulative_variance = np.cumsum(S ** 2) / np.sum(S ** 2)
    nd = np.argmax(cumulative_variance >= var) + 1
    W = Vh[:nd].T
    return W
