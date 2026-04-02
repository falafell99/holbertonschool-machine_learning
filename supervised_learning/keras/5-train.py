#!/usr/bin/env python3
"""Module to train a Keras model with validation data."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and validation data.
    network: the model to train.
    data: input data.
    labels: one-hot labels.
    batch_size: size of the batch.
    epochs: number of training epochs.
    validation_data: tuple of (X_valid, Y_valid) or None.
    verbose: boolean for printing output.
    shuffle: boolean for shuffling batches.
    Returns: the History object.
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
