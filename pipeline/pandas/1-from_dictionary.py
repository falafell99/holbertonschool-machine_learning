#!/usr/bin/env python3
"""Create DataFrame from dictionary."""

import pandas as pd

# Create dictionary with data
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create DataFrame from dictionary with row labels
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
