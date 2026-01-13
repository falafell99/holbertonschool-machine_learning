#!/usr/bin/env python3
"""Concatenate two DataFrames with specific conditions."""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Concatenate df2 (up to timestamp 1417411920) with df1.

    Args:
        df1: First DataFrame (coinbase)
        df2: Second DataFrame (bitstamp)

    Returns:
        Concatenated DataFrame with keys
    """
    # Set Timestamp as index for both dataframes
    df1 = index(df1)
    df2 = index(df2)

    # Select rows from df2 up to and including timestamp 1417411920
    df2_selected = df2[df2.index <= 1417411920]

    # Concatenate with keys
    result = pd.concat([df2_selected, df1], keys=['bitstamp', 'coinbase'])

    return result
