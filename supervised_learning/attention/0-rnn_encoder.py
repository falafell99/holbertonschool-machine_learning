#!/usr/bin/env python3
"""
Module for RNN Encoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder class to encode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab: An integer representing the size of the input vocabulary
            embedding: An integer representing the dimensionality of the
                embedding vector
            units: An integer representing the number of hidden units in
                the RNN cell
            batch: An integer representing the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros

        Returns:
            A tensor of shape (batch, units) containing the initialized
            hidden states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Executes the RNN encoder

        Args:
            x: A tensor of shape (batch, input_seq_len) containing the input
                to the encoder layer as word indices within the vocabulary
            initial: A tensor of shape (batch, units) containing the initial
                hidden state

        Returns:
            outputs: A tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder
            hidden: A tensor of shape (batch, units) containing the last
                hidden state of the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
