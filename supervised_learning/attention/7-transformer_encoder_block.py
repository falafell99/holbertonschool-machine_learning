#!/usr/bin/env python3
"""
Module for Transformer Encoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    EncoderBlock class to create an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        Args:
            dm: The dimensionality of the model
            h: The number of heads
            hidden: The number of hidden units in the fully connected layer
            drop_rate: The dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        Executes the Encoder Block

        Args:
            x: A tensor of shape (batch, input_seq_len, dm) containing the
                input to the encoder block
            training: A boolean to determine if the model is training
            mask: The mask to be applied for multi head attention

        Returns:
            A tensor of shape (batch, input_seq_len, dm) containing the
            block's output
        """
        # Sub-layer 1: Multi-Head Attention
        # Note: For self-attention in the encoder, Q, K, and V are all x
        attn_output, _ = self.mha(x, x, x, mask)

        # Apply dropout to the attention output
        attn_output = self.dropout1(attn_output, training=training)

        # Residual connection and Layer Normalization
        out1 = self.layernorm1(x + attn_output)

        # Sub-layer 2: Feed Forward Network
        # Pass through the hidden dense layer (with ReLU)
        hidden_output = self.dense_hidden(out1)

        # Pass through the output dense layer
        ffn_output = self.dense_output(hidden_output)

        # Apply dropout to the FFN output
        ffn_output = self.dropout2(ffn_output, training=training)

        # Residual connection and Layer Normalization
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
