#!/usr/bin/env python3
"""Visualize and transform DataFrame."""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set Date as index
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].fillna(method='ffill')
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 onwards
df = df[df.index >= '2017-01-01']

# Resample to daily intervals and aggregate
daily_df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

print(daily_df)

# Plotting (optional)
# daily_df[['High', 'Low', 'Open', 'Close']].plot(figsize=(12, 6))
# plt.title('Daily Bitcoin Prices (2017-2019)')
# plt.ylabel('Price (USD)')
# plt.show()
