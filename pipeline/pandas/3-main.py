#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
rename = __import__('3-rename').rename

# Create test data similar to the example
import pandas as pd
import numpy as np

# Create sample data with Timestamp and Close columns
dates = pd.date_range('2019-01-07 22:00:00', periods=6, freq='1min')
timestamps = dates.astype('int64') // 10**9  # Convert to Unix timestamp

test_data = pd.DataFrame({
    'Timestamp': timestamps,
    'Open': [4005.0, 4006.0, 4006.0, 4006.0, 4005.5, 4006.0],
    'High': [4006.0, 4006.0, 4006.0, 4006.0, 4006.0, 4006.0],
    'Low': [4005.0, 4005.0, 4005.0, 4005.0, 4005.0, 4005.0],
    'Close': [4005.01, 4006.01, 4006.01, 4006.01, 4005.50, 4005.99],
    'Volume_(BTC)': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'Volume_(Currency)': [4005.0, 4006.0, 4006.0, 4006.0, 4005.5, 4006.0],
    'Weighted_Price': [4005.5, 4006.0, 4006.0, 4006.0, 4005.75, 4006.0]
})

# Save to CSV for testing
test_data.to_csv('test_coinbase.csv', index=False)

# Test the function
df = from_file('test_coinbase.csv', ',')
df = rename(df)
print(df.tail())
