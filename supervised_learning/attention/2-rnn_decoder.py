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
        self.attention = SelfAttention(units)

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
        context, weights = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        concat_x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, s = self.gru(concat_x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)
        return y, s
