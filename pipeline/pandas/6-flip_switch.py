#!/usr/bin/env python3
"""Sort DataFrame in reverse order and transpose it."""


def flip_switch(df):
    """Sort data in reverse chronological order and transpose.

    Args:
        df: Input DataFrame

    Returns:
        Transformed DataFrame sorted and transposed
    """
    df_sorted = df.sort_index(ascending=False)
    return df_sorted.T
