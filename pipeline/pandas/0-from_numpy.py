#!/usr/bin/env python3
"""Create DataFrame from NumPy array."""

import pandas as pd


def from_numpy(array):
    """Create DataFrame with columns A, B, C, ...

    Args:
        array: NumPy array to convert

    Returns:
        pandas DataFrame with labeled columns
    """
    num_cols = array.shape[1] if len(array.shape) > 1 else 1
    columns = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)
