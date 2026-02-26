#!/usr/bin/env python3
"""Calculate F1 score for each class."""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculate F1 score for each class.

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent correct labels and
                  column indices represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing F1 score of each class
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)
    f1 = 2 * (prec * sens) / (prec + sens)
    return f1
