#!/usr/bin/env python3
"""Calculate specificity for each class."""
import numpy as np


def specificity(confusion):
    """Calculate specificity for each class.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent correct labels and
                  column indices represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing specificity of each class
    """
    classes = confusion.shape[0]
    specificity_values = np.zeros(classes)

    for i in range(classes):
        true_negatives = np.sum(confusion) - np.sum(confusion[i, :]) - np.sum(confusion[:, i]) + confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        denominator = true_negatives + false_positives
        specificity_values[i] = true_negatives / denominator

    return specificity_values
