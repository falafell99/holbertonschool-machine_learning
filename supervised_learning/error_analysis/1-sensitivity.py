#!/usr/bin/env python3
"""Calculate sensitivity for each class."""
import numpy as np


def sensitivity(confusion):
    """Calculate sensitivity for each class.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent correct labels and
                  column indices represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing sensitivity of each class
    """
    classes = confusion.shape[0]
    sensitivity_values = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i][i]
        false_negatives = np.sum(confusion[i]) - true_positives
        sensitivity_values[i] = true_positives / (true_positives + false_negatives)

    return sensitivity_values
