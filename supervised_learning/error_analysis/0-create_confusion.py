#!/usr/bin/env python3
"""Create confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create confusion matrix.

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes) with correct labels
        logits: one-hot numpy.ndarray of shape (m, classes) with predicted labels

    Returns:
        confusion: numpy.ndarray of shape (classes, classes) with row indices
                  representing correct labels and column indices representing
                  predicted labels
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))

    for i in range(m):
        true_class = np.argmax(labels[i])
        pred_class = np.argmax(logits[i])
        confusion[true_class][pred_class] += 1

    return confusion
