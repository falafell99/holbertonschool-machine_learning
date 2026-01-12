#!/usr/bin/env python3
"""Slice DataFrame by selecting every 60th row of specific columns."""


def slice(df):
    """Extract High, Low, Close, Volume_(BTC) and select every 60th row.

    Args:
        df: Input DataFrame

    Returns:
        Sliced DataFrame with every 60th row
    """
    selected = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    return selected.iloc[::60]
