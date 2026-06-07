#!/usr/bin/env python3
"""Module that contains the sparse autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes for each
            hidden layer in the encoder, respectively. The hidden layers
            should be reversed for the decoder.
        latent_dims (int): The dimensions of the latent space representation.
        lambtha (float): The regularization parameter used for L1
            regularization on the encoded output.

    Returns:
        tuple: (encoder, decoder, auto)
        - encoder is the encoder model
        - decoder is the decoder model
        - auto is the sparse autoencoder model compiled with adam
          optimization and binary cross-entropy loss.
    """
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    reg = keras.regularizers.L1(lambtha)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=reg
    )(encoded)

    encoder = keras.Model(inputs=inputs, outputs=latent)

    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    # Вот тут главное изменение: передаем саму функцию вместо строки!
    auto.compile(
        optimizer='adam',
        loss=keras.losses.binary_crossentropy
    )

    return encoder, decoder, auto
