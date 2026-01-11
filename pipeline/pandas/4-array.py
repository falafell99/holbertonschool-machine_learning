#!/usr/bin/env python3
"""Convert last 10 rows of High and Close columns to numpy array."""

import pandas as pd


def array(df):
    """Select last 10 rows of High and Close and convert to numpy array."""
    # Select last 10 rows of High and Close columns
    selected = df[['High', 'Close']].tail(10)
    
    # Convert to numpy array
    return selected.to_numpy()
