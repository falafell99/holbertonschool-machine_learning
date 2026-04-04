#!/usr/bin/env python3
"""Module to perform a strided convolution on grayscale images."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with strides and padding.
    images: numpy.ndarray (m, h, w) with multiple grayscale images.
    kernel: numpy.ndarray (kh, kw) containing the kernel.
    padding: tuple (ph, pw), 'same', or 'valid'.
    stride: tuple (sh, sw).
    Returns: numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Определяем ph и pw в зависимости от типа паддинга
    if padding == 'same':
        # Формула для "same" с учетом шага (stride)
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Добавляем паддинг
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Вычисляем выходные размеры с учетом шага
    h_out = ((h + 2 * ph - kh) // sh) + 1
    w_out = ((w + 2 * pw - kw) // sw) + 1

    # Инициализируем выходной массив
    convolved = np.zeros((m, h_out, w_out))

    # Циклы по высоте и ширине ВЫХОДНОГО изображения
    for i in range(h_out):
        for j in range(w_out):
            # Индексы входа сдвигаются на шаг (sh, sw)
            # i*sh и j*sw — это верхний левый угол окна свертки
            convolved[:, i, j] = np.sum(
                images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel,
                axis=(1, 2)
            )

    return convolved
