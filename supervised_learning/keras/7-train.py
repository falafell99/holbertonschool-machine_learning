#!/usr/bin/env python3
"""Module to train a Keras model with learning rate decay."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with learning rate decay.
    network: the model to train.
    data: input data.
    labels: one-hot labels.
    batch_size: size of the batch.
    epochs: number of epochs.
    validation_data: data for validation.
    early_stopping: whether to use early stopping.
    patience: patience for early stopping.
    learning_rate_decay: whether to use inverse time decay.
    alpha: initial learning rate.
    decay_rate: decay rate.
    verbose: printing output.
    shuffle: shuffling batches.
    Returns: the History object.
    """
    def scheduler(epoch):
        """Inverse time decay function."""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            ))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(
                scheduler,
                verbose=1
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
