#!/usr/bin/env python3
"""
Module for Scaled Dot Product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    Args:
        Q: A tensor with its last two dimensions as (..., seq_len_q, dk)
            containing the query matrix
        K: A tensor with its last two dimensions as (..., seq_len_v, dk)
            containing the key matrix
        V: A tensor with its last two dimensions as (..., seq_len_v, dv)
            containing the value matrix
        mask: A tensor that can be broadcast into
            (..., seq_len_q, seq_len_v) containing the optional mask, or
            defaulted to None

    Returns:
        output: A tensor with its last two dimensions as (..., seq_len_q, dv)
            containing the scaled dot product attention
        weights: A tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v) containing the attention weights
    """
    # Calculate Q * K^T
    # Q shape: (..., seq_len_q, dk)
    # K transpose shape: (..., dk, seq_len_v)
    # matmul_qk shape: (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale matmul_qk
    # Cast dk to float32 for mathematical operations
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax on the last axis (seq_len_v) to get probabilities
    # weights shape: (..., seq_len_q, seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply weights by V
    # weights shape: (..., seq_len_q, seq_len_v)
    # V shape: (..., seq_len_v, dv)
    # output shape: (..., seq_len_q, dv)
    output = tf.matmul(weights, V)

    return output, weights
