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
        # Load dataset using the provided setup script instead of tfds
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
        # Use HuggingFace transformers as per updated ALX curriculum
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True,
            clean_up_tokenization_spaces=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True,
            clean_up_tokenization_spaces=True
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        """
        # Convert tensors to string (transformers expect Python strings)
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Encode without adding special [CLS]/[SEP] tokens
        pt_encoded = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False)
        en_encoded = self.tokenizer_en.encode(
            en_text, add_special_tokens=False)

        # Add vocab_size as start token and vocab_size + 1 as end token
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
            self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
