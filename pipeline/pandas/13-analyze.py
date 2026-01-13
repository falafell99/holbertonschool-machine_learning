#!/usr/bin/env python3
"""Compute descriptive statistics for DataFrame."""


def analyze(df):
    """Compute descriptive statistics for all columns except Timestamp.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with descriptive statistics
    """
    # Drop Timestamp column if it exists
    if 'Timestamp' in df.columns:
        df_without_timestamp = df.drop(columns=['Timestamp'])
    else:
        df_without_timestamp = df

    # Compute descriptive statistics
    stats = df_without_timestamp.describe()

    return stats
