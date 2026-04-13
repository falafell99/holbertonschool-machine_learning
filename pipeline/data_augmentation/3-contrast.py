#!/usr/bin/env python3
"""Module to adjust image contrast."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.
    image: a 3D tf.Tensor containing the image to adjust.
    lower: float, lower bound for the random contrast factor.
    upper: float, upper bound for the random contrast factor.
    Returns: the contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower, upper)
