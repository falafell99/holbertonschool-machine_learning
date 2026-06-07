#!/usr/bin/env python3
"""Module that contains the convolutional autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.

    Args:
        input_dims (tuple): A tuple of integers containing the dimensions
            of the model input.
        filters (list): A list containing the number of filters for each
            convolutional layer in the encoder, respectively. The filters
            should be reversed for the decoder.
        latent_dims (tuple): A tuple of integers containing the dimensions
            of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
        - encoder is the encoder model
        - decoder is the decoder model
        - auto is the full autoencoder model compiled with adam optimization
          and binary cross-entropy loss.
    """
    # --- Encoder ---
    inputs = keras.Input(shape=input_dims)
    encoded = inputs

    for f in filters:
        encoded = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(encoded)
        encoded = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(encoded)

    encoder = keras.Model(inputs=inputs, outputs=encoded)

    # --- Decoder ---
    latent_inputs = keras.Input(shape=latent_dims)
    decoded = latent_inputs

    rev_filters = list(reversed(filters))

    # Iterate through reversed filters
    for i, f in enumerate(rev_filters):
        if i == len(rev_filters) - 1:
            # The second to last convolution (last one in this loop)
            decoded = keras.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                activation='relu',
                padding='valid'
            )(decoded)
        else:
            decoded = keras.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                activation='relu',
                padding='same'
            )(decoded)

        decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)

    # The last convolution
    decoded = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )(decoded)

    decoder = keras.Model(inputs=latent_inputs, outputs=decoded)

    # --- Autoencoder ---
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
