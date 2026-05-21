#!/usr/bin/env python3
"""Module to initialize t-SNE variables"""
import numpy as np


def P_init(X, perplexity):
    """Initializes variables to calculate the P affinities in t-SNE"""
    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
