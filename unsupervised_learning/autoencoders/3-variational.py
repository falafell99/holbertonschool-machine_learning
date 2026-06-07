#!/usr/bin/env python3
"""Module that contains the variational autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes for each
            hidden layer in the encoder. Should be reversed for the decoder.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
        - encoder: outputs the latent representation, mean, and log variance.
        - decoder: the decoder model.
        - auto: the full autoencoder model compiled with adam optimization
          and binary cross-entropy loss.
    """
    # --- Encoder ---
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)

    def sampling(args):
        """Samples from the latent space."""
        z_m, z_l_v = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(0.5 * z_l_v) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    # Encoder returns latent representation, mean, and log variance
    encoder = keras.Model(inputs=inputs, outputs=[z, z_mean, z_log_var])

    # --- Decoder ---
    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # --- Autoencoder ---
    # We pass the input through the encoder, but only take `z` (index 0)
    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    # Add Kullback-Leibler divergence loss to the autoencoder
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean) -
        keras.backend.exp(z_log_var),
        axis=-1
    )
    auto.add_loss(keras.backend.mean(kl_loss))

    # Compile with specified optimizer and reconstruction loss
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
