#!/usr/bin/env python3
"""
Module for RNN Decoder
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab: An integer representing the size of the output vocabulary
            embedding: An integer representing the dimensionality of the
                embedding vector
            units: An integer representing the number of hidden units in
                the RNN cell
            batch: An integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
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
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Executes the RNN decoder

        Args:
            x: A tensor of shape (batch, 1) containing the previous word in
                the target sequence as an index of the target vocabulary
            s_prev: A tensor of shape (batch, units) containing the
                previous decoder hidden state
            hidden_states: A tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder

        Returns:
            y: A tensor of shape (batch, vocab) containing the output word
                as a one hot vector in the target vocabulary
            s: A tensor of shape (batch, units) containing the new decoder
                hidden state
        """
        # Instantiate SelfAttention locally as intended by the task description
        attention = SelfAttention(s_prev.shape[1])

        # Calculate context vector and attention weights
        context, weights = attention(s_prev, hidden_states)

        # Pass x through the embedding layer
        x = self.embedding(x)

        # Expand the context vector to have shape (batch, 1, units)
        context = tf.expand_dims(context, 1)

        # Concatenate context vector with x in that order
        concat_x = tf.concat([context, x], axis=-1)

        # Pass the concatenated vector to the GRU
        y, s = self.gru(concat_x, initial_state=s_prev)

        # Reshape the output to (batch, units) BEFORE passing to Dense layer
        y = tf.reshape(y, (-1, y.shape[2]))

        # Pass the reshaped output to the Dense layer F
        y = self.F(y)

        return y, s
