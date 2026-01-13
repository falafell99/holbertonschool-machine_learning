#!/usr/bin/env python3
"""Remove rows where Close has NaN values."""


def prune(df):
    """Remove entries where Close has NaN values.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with NaN values removed from Close column
    """
    return df.dropna(subset=['Close'])
