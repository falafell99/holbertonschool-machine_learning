#!/usr/bin/env python3
"""Convert last 10 rows of High and Close columns to numpy array."""


def array(df):
    """Select last 10 rows of High and Close and convert to numpy array.
    
    Args:
        df: DataFrame with High and Close columns
        
    Returns:
        numpy.ndarray with last 10 rows of High and Close
    """
    selected = df[['High', 'Close']].tail(10)
    return selected.to_numpy()
