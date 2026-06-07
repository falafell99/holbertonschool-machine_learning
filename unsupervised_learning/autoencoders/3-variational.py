#!/usr/bin/env python3
"""Module that contains the variational autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder."""
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    mu = keras.layers.Dense(latent_dims, activation=None)(encoded)
    log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)

    def sampling(args):
        mu_, log_var_ = args
        eps = keras.backend.random_normal(shape=keras.backend.shape(mu_))
        return mu_ + keras.backend.exp(log_var_ / 2) * eps

    z = keras.layers.Lambda(sampling)([mu, log_var])
    encoder = keras.Model(inputs=inputs, outputs=[z, mu, log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    def vae_loss(x, x_decoded):
        rec = keras.losses.binary_crossentropy(x, x_decoded) * input_dims
        kl = -0.5 * keras.backend.sum(
            1 + log_var - keras.backend.square(mu)
            - keras.backend.exp(log_var), axis=1)
        return keras.backend.mean(rec + kl)

    auto.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, auto
