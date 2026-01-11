#!/usr/bin/env python3
"""Rename Timestamp column and convert to datetime."""

import pandas as pd


def rename(df):
    """Rename Timestamp column and convert to datetime.
    
    Args:
        df: DataFrame with Timestamp column
        
    Returns:
        DataFrame with Datetime column and only Close column
    """
    df = df.copy()
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
