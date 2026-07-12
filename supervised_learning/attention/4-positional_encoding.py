#!/usr/bin/env python3
"""
Module for Positional Encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    Args:
        max_seq_len: An integer representing the maximum sequence length
        dm: The model depth (dimensionality)

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    # Create arrays for positions and dimensions
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    # Calculate the angle rates: 1 / (10000 ^ (2i / dm))
    # We use (2 * (i // 2)) to ensure that both the sine (even indices) and
    # cosine (odd indices) for a given dimension pair have the same frequency.
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))

    # Multiply positions by angle rates
    angle_rads = pos * angle_rates

    # Apply sin to even indices in the array (2i)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array (2i+1)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
