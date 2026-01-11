#!/usr/bin/env python3
"""Rename Timestamp column and convert to datetime."""

import pandas as pd


def rename(df):
    """
    Rename Timestamp column and convert to datetime.
    
    Args:
        df: DataFrame with Timestamp column
        
    Returns:
        DataFrame with Datetime column and only Close column
    """
    # Copy to avoid modifying original
    df = df.copy()
    
    # Rename Timestamp to Datetime
    df = df.rename(columns={'Timestamp': 'Datetime'})
    
    # Convert Unix timestamp to datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    
    # Select only Datetime and Close columns
    df = df[['Datetime', 'Close']]
    
    return df
