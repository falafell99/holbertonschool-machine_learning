#!/usr/bin/env python3
"""Module to perform a convolution on images with channels."""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.
    images: numpy.ndarray (m, h, w, c) containing multiple images.
    kernel: numpy.ndarray (kh, kw, c) containing the kernel.
    padding: tuple (ph, pw), 'same', or 'valid'.
    stride: tuple (sh, sw).
    Returns: numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Паддинг добавляем только к высоте (ось 1) и ширине (ось 2)
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    h_out = ((h + 2 * ph - kh) // sh) + 1
    w_out = ((w + 2 * pw - kw) // sw) + 1

    convolved = np.zeros((m, h_out, w_out))

    # Снова 2 цикла по выходным координатам
    for i in range(h_out):
        for j in range(w_out):
            # Слайс теперь 4D: (все_изображения, высота, ширина, все_каналы)
            # Суммируем по осям 1, 2 и 3 (высота, ширина, каналы)
            convolved[:, i, j] = np.sum(
                images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] * kernel,
                axis=(1, 2, 3)
            )

    return convolved
