#!/usr/bin/env python3
"""Module to preprocess raw BTC data for RNN forecasting."""
import sys
import pandas as pd


def preprocess_data(input_path, output_path):
    """
    Cleans and resamples raw BTC data.

    Args:
        input_path (str): The path to the raw 60-second BTC csv.
        output_path (str): The path to save the preprocessed hourly data.
    """
    print("Loading raw data...")
    df = pd.read_csv(input_path)

    # Drop missing values
    df = df.dropna()

    # Convert Unix timestamp to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # Resample from 1-minute to 1-hour windows
    print("Resampling data to 1-hour windows...")
    df_hourly = df.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })

    # Drop any new NaNs introduced by empty hours
    df_hourly = df_hourly.dropna()

    # Save to CSV
    df_hourly.to_csv(output_path)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: ./preprocess_data.py <input_csv> <output_csv>")
        sys.exit(1)

    preprocess_data(sys.argv[1], sys.argv[2])
