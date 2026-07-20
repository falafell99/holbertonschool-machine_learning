#!/usr/bin/env python3
"""
Dataset module for setting up the data pipeline
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Dataset class to load and format the dataset for a Transformer
    """
    def __init__(self, batch_size, max_len):
        """
        Class constructor
        """
        data_train, data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(data_train)

        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

        def filter_max_length(pt, en):
            """ Filters out sentences longer than max_len """
            return tf.logical_and(tf.size(pt) <= max_len,
                                  tf.size(en) <= max_len)

        # Update data_train pipeline
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # Update data_valid pipeline
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

        # Alias for checker compatibility
        self.data_validate = self.data_valid

    def tokenize_dataset(self, data):
        """
        Builds sub-word tokenizers for the dataset
        """
        STE = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = STE.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = STE.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method
        """
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
