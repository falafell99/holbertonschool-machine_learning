#!/usr/bin/env python3
"""Module for Batch Normalization."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch norm.
    Z: numpy.ndarray of shape (m, n) to be normalized.
    gamma: numpy.ndarray of shape (1, n) containing the scales.
    beta: numpy.ndarray of shape (1, n) containing the offsets.
    epsilon: small number used to avoid division by zero.
    Returns: the normalized Z matrix.
    """
    # Вычисляем среднее и дисперсию по строкам (для каждой фичи)
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Стандартизируем Z
    Z_centered = Z - mean
    Z_norm = Z_centered / np.sqrt(variance + epsilon)

    # Применяем масштабирование и смещение (Scale and Shift)
    return gamma * Z_norm + beta
