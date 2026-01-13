#!/usr/bin/env python3
"""Fill missing values in DataFrame."""


def fill(df):
    """Fill missing values according to specifications.

    Args:
        df: Input DataFrame

    Returns:
        Modified DataFrame with filled values
    """
    # Remove Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Fill Close column with previous row's value (forward fill)
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill High, Low, Open with corresponding Close value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Fill Volume columns with 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
