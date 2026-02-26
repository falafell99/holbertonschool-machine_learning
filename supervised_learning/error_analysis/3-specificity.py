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
        tp = confusion[i, i]
        fn = np.sum(confusion[i, :]) - tp
        fp = np.sum(confusion[:, i]) - tp
        tn = np.sum(confusion) - tp - fn - fp
        denominator = tn + fp
        specificity_values[i] = tn / denominator

    return specificity_values
