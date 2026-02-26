# Error Analysis

This directory contains implementations of error analysis techniques for machine learning.

## Files

### 0-create_confusion.py
Contains function to create a confusion matrix.

**Function: create_confusion_matrix(labels, logits)**
- Creates a confusion matrix from one-hot encoded labels and predictions
- Args:
  - labels: one-hot numpy.ndarray of shape (m, classes) with correct labels
  - logits: one-hot numpy.ndarray of shape (m, classes) with predicted labels
- Returns:
  - confusion matrix of shape (classes, classes) with row indices representing
    correct labels and column indices representing predicted labels

## Requirements
- Python 3.5+
- pycodestyle 2.5
- numpy
