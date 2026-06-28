# Bitcoin (BTC) Time Series Forecasting

This project uses Recurrent Neural Networks (RNNs) in TensorFlow/Keras to forecast the price of Bitcoin.

## Structure
* `preprocess_data.py`: Cleans and resamples raw minute-by-minute BTC data into 1-hour windows.
* `forecast_btc.py`: Builds and trains an LSTM network to predict the next hour's closing price based on the past 24 hours.
