#!/usr/bin/env python3
"""Module to perform a valid convolution on grayscale images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.
    images: numpy.ndarray (m, h, w) with multiple grayscale images.
    kernel: numpy.ndarray (kh, kw) containing the kernel.
    Returns: numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Вычисляем размер выходного изображения
    h_out = h - kh + 1
    w_out = w - kw + 1

    # Инициализируем массив нулями
    convolved = np.zeros((m, h_out, w_out))

    # Используем только 2 цикла по высоте и ширине выхода
    for i in range(h_out):
        for j in range(w_out):
            # Извлекаем слайс из ВСЕХ изображений сразу (векторизация)
            # Умножаем на ядро и суммируем по осям высоты и ширины (1, 2)
            convolved[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return convolved
