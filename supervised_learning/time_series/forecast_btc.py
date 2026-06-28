#!/usr/bin/env python3
"""Module that creates, trains, and validates a keras model for BTC."""
import pandas as pd
import tensorflow as tf


def create_tf_dataset(data, targets, seq_length, batch_size):
    """Creates a tf.data.Dataset for time series forecasting."""
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=data[:-1],
        targets=targets[seq_length:],
        sequence_length=seq_length,
        batch_size=batch_size,
        shuffle=True
    )
    return dataset


def forecast_btc():
    """Builds, trains and validates an RNN model for BTC forecasting."""
    # 1. Load preprocessed hourly data
    df = pd.read_csv('preprocessed_btc.csv', index_col='Timestamp')
    close_idx = df.columns.get_loc('Close')
    data = df.values

    # 2. Split data: 70% Train, 20% Validation, 10% Test
    n = len(data)
    train_df = data[0:int(n * 0.7)]
    val_df = data[int(n * 0.7):int(n * 0.9)]
    test_df = data[int(n * 0.9):]

    # 3. Normalize the data
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # 4. Create tf.data.Datasets
    seq_length = 24
    batch_size = 256

    train_ds = create_tf_dataset(
        train_df, train_df[:, close_idx], seq_length, batch_size)
    val_ds = create_tf_dataset(
        val_df, val_df[:, close_idx], seq_length, batch_size)
    test_ds = create_tf_dataset(
        test_df, test_df[:, close_idx], seq_length, batch_size)

    # 5. Build the RNN (LSTM) Architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, data.shape[1])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # 6. Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    model.summary()

    # 7. Train the model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("Training the model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stopping]
    )

    # 8. Evaluate on test set
    print("Evaluating on test dataset...")
    test_loss, test_mae = model.evaluate(test_ds)
    print(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    return model


if __name__ == '__main__':
    forecast_btc()
