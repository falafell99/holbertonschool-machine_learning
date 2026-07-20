#!/usr/bin/env python3
"""
Module for Multi Head Attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class to perform multi-head attention
    """

    def __init__(self, dm, h):
        """
        Class constructor

        Args:
            dm: An integer representing the dimensionality of the model
            h: An integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        Transpose the result such that the shape is
        (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Executes the Multi Head Attention layer

        Args:
            Q: A tensor of shape (batch, seq_len_q, dk) containing the input
                to generate the query matrix
            K: A tensor of shape (batch, seq_len_v, dk) containing the input
                to generate the key matrix
            V: A tensor of shape (batch, seq_len_v, dv) containing the input
                to generate the value matrix
            mask: Is always None

        Returns:
            output: A tensor with its last two dimensions as
                (..., seq_len_q, dm) containing the scaled dot product attention
            weights: A tensor with its last three dimensions as
                (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Linear transformations for Q, K, V
        # Q shape: (batch, seq_len_q, dm)
        # K shape: (batch, seq_len_v, dm)
        # V shape: (batch, seq_len_v, dm)
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split heads
        # Q shape: (batch_size, h, seq_len_q, depth)
        # K shape: (batch_size, h, seq_len_v, depth)
        # V shape: (batch_size, h, seq_len_v, depth)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Calculate scaled dot product attention
        # scaled_attention shape: (batch_size, h, seq_len_q, depth)
        # attention_weights shape: (batch_size, h, seq_len_q, seq_len_v)
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Transpose and reshape back to original
        # Transpose shape: (batch_size, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concat shape: (batch_size, seq_len_q, dm)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # Pass the concatenated vectors through the final dense layer
        # output shape: (batch_size, seq_len_q, dm)
        output = self.linear(concat_attention)

        return output, weights
