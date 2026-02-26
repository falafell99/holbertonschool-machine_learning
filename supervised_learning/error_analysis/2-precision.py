#!/usr/bin/env python3
"""Calculate precision for each class."""
import numpy as np


def precision(confusion):
    """Calculate precision for each class.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent correct labels and
                  column indices represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing precision of each class
    """
    classes = confusion.shape[0]
    precision_values = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i][i]
        false_positives = np.sum(confusion[:, i]) - true_positives
        denominator = true_positives + false_positives
        precision_values[i] = true_positives / denominator

    return precision_values
