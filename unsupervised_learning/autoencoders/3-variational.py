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

    z = keras.layers.Lambda(
        lambda args: args[0] + keras.backend.exp(args[1] / 2) *
        keras.backend.random_normal(
            shape=(keras.backend.shape(args[0])[0], latent_dims)
        )
    )([mu, log_var])

    encoder = keras.Model(inputs=inputs, outputs=[z, mu, log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    auto_inputs = keras.Input(shape=(input_dims,))
    z_out, mu_out, log_var_out = encoder(auto_inputs)
    auto_outputs = decoder(z_out)
    auto = keras.Model(inputs=auto_inputs, outputs=auto_outputs)

    rec_loss = keras.backend.mean(
        keras.losses.binary_crossentropy(auto_inputs, auto_outputs)
    ) * input_dims
    kl_loss = -0.5 * keras.backend.mean(
        keras.backend.sum(
            1 + log_var_out - keras.backend.square(mu_out)
            - keras.backend.exp(log_var_out), axis=1
        )
    )
    auto.add_loss(rec_loss + kl_loss)
    auto.compile(optimizer='adam')
    return encoder, decoder, auto
