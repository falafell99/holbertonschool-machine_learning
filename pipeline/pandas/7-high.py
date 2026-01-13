#!/usr/bin/env python3
"""Sort DataFrame by High price in descending order."""


def high(df):
    """Sort DataFrame by High price in descending order.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame sorted by High price descending
    """
    return df.sort_values(by='High', ascending=False)
