#!/usr/bin/env python3
"""
Module for Transformer Decoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Decoder class to create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N: The number of blocks in the encoder
            dm: The dimensionality of the model
            h: The number of heads
            hidden: The number of hidden units in the fully connected layer
            target_vocab: The size of the target vocabulary
            max_seq_len: The maximum sequence length possible
            drop_rate: The dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab,
            output_dim=dm
        )
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Executes the Decoder

        Args:
            x: A tensor of shape (batch, target_seq_len) containing the
                input to the decoder
            encoder_output: A tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            training: A boolean to determine if the model is training
            look_ahead_mask: The mask to be applied to the first multi
                head attention layer
            padding_mask: The mask to be applied to the second multi
                head attention layer

        Returns:
            A tensor of shape (batch, target_seq_len, dm) containing the
            decoder output
        """
        seq_len = tf.shape(x)[1]

        # Pass x through the embedding layer
        x = self.embedding(x)

        # Scale the embeddings by multiplying by the square root of dm
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Cast positional encoding to tensor and slice to seq_len
        pos_encoding = tf.constant(self.positional_encoding,
                                   dtype=tf.float32)
        pos_encoding = pos_encoding[tf.newaxis, :seq_len, :]

        # Add positional encoding to embeddings
        x += pos_encoding

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each DecoderBlock N times
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)

        return x
