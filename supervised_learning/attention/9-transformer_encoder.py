#!/usr/bin/env python3
"""
Module for Transformer Encoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class to create the encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N: The number of blocks in the encoder
            dm: The dimensionality of the model
            h: The number of heads
            hidden: The number of hidden units in the fully connected layer
            input_vocab: The size of the input vocabulary
            max_seq_len: The maximum sequence length possible
            drop_rate: The dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab,
            output_dim=dm
        )
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Executes the Encoder

        Args:
            x: A tensor of shape (batch, input_seq_len) containing the
                input to the encoder
            training: A boolean to determine if the model is training
            mask: The mask to be applied for multi head attention

        Returns:
            A tensor of shape (batch, input_seq_len, dm) containing the
            encoder output
        """
        seq_len = tf.shape(x)[1]

        # Pass x through the embedding layer
        x = self.embedding(x)

        # Scale the embeddings by multiplying by the square root of dm
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Cast positional encoding to tensor and slice to seq_len
        # Adding tf.newaxis ensures robust broadcasting
        # over the batch dimension
        pos_encoding = tf.constant(self.positional_encoding,
                                   dtype=tf.float32)
        pos_encoding = pos_encoding[tf.newaxis, :seq_len, :]

        # Add positional encoding to embeddings
        x += pos_encoding

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each EncoderBlock N times
        for block in self.blocks:
            x = block(x, training, mask)

        return x
