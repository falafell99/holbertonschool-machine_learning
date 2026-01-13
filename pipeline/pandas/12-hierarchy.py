#!/usr/bin/env python3
"""Concatenate and rearrange hierarchical index."""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """Rearrange MultiIndex with Timestamp as first level.

    Args:
        df1: First DataFrame (coinbase)
        df2: Second DataFrame (bitstamp)

    Returns:
        Concatenated DataFrame with rearranged index
    """
    # Set Timestamp as index
    df1 = index(df1)
    df2 = index(df2)

    # Select rows between timestamps 1417411980 and 1417417980 (inclusive)
    mask1 = (df1.index >= 1417411980) & (df1.index <= 1417417980)
    mask2 = (df2.index >= 1417411980) & (df2.index <= 1417417980)

    df1_selected = df1[mask1]
    df2_selected = df2[mask2]

    # Concatenate with keys
    concatenated = pd.concat([df2_selected, df1_selected],
                             keys=['bitstamp', 'coinbase'])

    # Rearrange MultiIndex: Timestamp as first level
    concatenated = concatenated.swaplevel(0, 1).sort_index(level=0)

    return concatenated
