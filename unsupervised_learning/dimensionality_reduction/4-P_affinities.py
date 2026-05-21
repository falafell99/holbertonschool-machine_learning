#!/usr/bin/env python3
"""Module to calculate symmetric P affinities of a data set"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set
    """
    n, d = X.shape
    D, P, betas, H_target = P_init(X, perplexity)

    for i in range(n):
        # Exclude the point's distance to itself
        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        beta = betas[i, 0]
        beta_min = None
        beta_max = None

        Hi, Pi = HP(Di, beta)
        H_diff = Hi - H_target

        # Binary search for beta
        while np.abs(H_diff) > tol:
            if H_diff > 0:
                # Entropy too high -> decrease variance -> increase beta
                beta_min = beta
                if beta_max is None:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                # Entropy too low -> increase variance -> decrease beta
                beta_max = beta
                if beta_min is None:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

            Hi, Pi = HP(Di, beta)
            H_diff = Hi - H_target

        betas[i, 0] = beta
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    # Calculate symmetric P affinities
    P = (P + P.T) / (2 * n)

    return P
