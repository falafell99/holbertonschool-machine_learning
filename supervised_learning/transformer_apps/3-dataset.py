#!/usr/bin/env python3
"""
Dataset module for setting up the data pipeline
"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """
    Dataset class to load and format the dataset for a Transformer
    """
    def __init__(self, batch_size, max_len):
        """
        Class constructor
        """
        # Load dataset using the provided setup script
        data_train, data_valid = load_pt2en()

        # Build tokenizers using transformers
        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(data_train)

        # Map the encoding function over the datasets
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
        # Load a base tokenizer (cached locally in ALX environment)
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True
        )

        # Create iterators using batches to make training lightning fast
        def pt_iterator():
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [s.decode('utf-8') for s in pt]

        def en_iterator():
            for _, en in data.batch(1000).as_numpy_iterator():
                yield [s.decode('utf-8') for s in en]

        # Train new tokenizers from iterators with max vocab size 2**13 (8192)
        vocab_size = 2 ** 13
        tokenizer_pt = base_tokenizer.train_new_from_iterator(
            pt_iterator(), vocab_size
        )
        tokenizer_en = base_tokenizer.train_new_from_iterator(
            en_iterator(), vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        """
        # Convert tensors to strings
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Encode text using the newly trained tokenizers
        pt_encoded = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_encoded = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        # Prepend vocab_size (start) and append vocab_size + 1 (end)
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            pt_encoded + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + \
            en_encoded + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method
        """
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
