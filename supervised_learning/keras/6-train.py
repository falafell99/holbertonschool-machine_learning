#!/usr/bin/env python3
"""Module to train a Keras model with Early Stopping."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and early stopping.
    network: the model to train.
    data: input data.
    labels: one-hot labels.
    batch_size: size of the batch.
    epochs: number of epochs.
    validation_data: tuple of (X_valid, Y_valid).
    early_stopping: boolean, whether to use early stopping.
    patience: patience for early stopping.
    verbose: boolean for printing output.
    shuffle: boolean for shuffling batches.
    Returns: the History object.
    """
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
    return history
