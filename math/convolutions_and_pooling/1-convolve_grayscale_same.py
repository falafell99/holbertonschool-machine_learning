#!/usr/bin/env python3
"""Module to perform a same convolution on grayscale images."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.
    images: numpy.ndarray (m, h, w) with multiple grayscale images.
    kernel: numpy.ndarray (kh, kw) containing the kernel.
    Returns: numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Для "same" свертки паддинг вычисляется так, чтобы h_out = h
    # Формула: p = (k - 1) / 2. Если k четное, берем верхнюю границу.
    ph = kh // 2
    pw = kw // 2

    # Добавляем паддинг (только по высоте и ширине)
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Инициализируем выходной массив (размер остается h x w)
    convolved = np.zeros((m, h, w))

    # Используем 2 цикла по высоте и ширине
    for i in range(h):
        for j in range(w):
            # Извлекаем слайс из дополненного изображения
            # Размер слайса всегда равен размеру ядра (kh, kw)
            convolved[:, i, j] = np.sum(
                images_padded[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return convolved
