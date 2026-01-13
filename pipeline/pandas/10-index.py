#!/usr/bin/env python3
"""Set Timestamp column as index."""


def index(df):
    """Set Timestamp column as index of DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with Timestamp as index
    """
    return df.set_index('Timestamp')
