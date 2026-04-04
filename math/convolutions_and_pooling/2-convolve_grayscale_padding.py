#!/usr/bin/env python3
"""Module to perform a convolution with custom padding."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.
    images: numpy.ndarray (m, h, w) with multiple grayscale images.
    kernel: numpy.ndarray (kh, kw) containing the kernel.
    padding: tuple of (ph, pw).
    Returns: numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Добавляем указанный паддинг (ph по высоте, pw по ширине)
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Вычисляем выходные размеры по формуле:
    # H_out = H + 2*Ph - Kh + 1
    h_out = h + (2 * ph) - kh + 1
    w_out = w + (2 * pw) - kw + 1

    # Инициализируем выходной массив
    convolved = np.zeros((m, h_out, w_out))

    # Используем только 2 разрешенных цикла (по высоте и ширине выхода)
    for i in range(h_out):
        for j in range(w_out):
            # Векторизованная свертка по всем изображениям сразу
            convolved[:, i, j] = np.sum(
                images_padded[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return convolved
