#!/usr/bin/env python3
"""Module that contains the sparse autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder."""
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
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
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
