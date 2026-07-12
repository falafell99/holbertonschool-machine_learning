#!/usr/bin/env python3
"""
Module for Self Attention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention class to calculate the attention for machine translation
    based on Bahdanau Attention
    """

    def __init__(self, units):
        """
        Class constructor

        Args:
            units: An integer representing the number of hidden units in the
                alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Executes the Self Attention layer

        Args:
            s_prev: A tensor of shape (batch, units) containing the previous
                decoder hidden state
            hidden_states: A tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder

        Returns:
            context: A tensor of shape (batch, units) that contains the
                context vector for the decoder
            weights: A tensor of shape (batch, input_seq_len, 1) that
                contains the attention weights
        """
        # Expand s_prev to have shape (batch, 1, units) so we can add it
        # to hidden_states (batch, input_seq_len, units)
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Calculate the score (Bahdanau alignment score)
        # score shape == (batch, input_seq_len, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) +
                                  self.U(hidden_states)))

        # Calculate the attention weights
        # weights shape == (batch, input_seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)

        # Calculate the context vector
        # context shape after multiplication == (batch, input_seq_len, units)
        context = weights * hidden_states

        # context shape after sum == (batch, units)
        context = tf.reduce_sum(context, axis=1)

        return context, weights
