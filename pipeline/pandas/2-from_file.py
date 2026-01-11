#!/usr/bin/env python3
"""Load data from file into DataFrame."""

import pandas as pd


def from_file(filename, delimiter):
    """Load data from file."""
    return pd.read_csv(filename, delimiter=delimiter)
