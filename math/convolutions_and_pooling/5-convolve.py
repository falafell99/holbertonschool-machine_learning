#!/usr/bin/env python3
"""Module to perform a convolution with multiple kernels."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.
    images: numpy.ndarray (m, h, w, c) containing multiple images.
    kernels: numpy.ndarray (kh, kw, c, nc) containing the kernels.
    padding: tuple (ph, pw), 'same', or 'valid'.
    stride: tuple (sh, sw).
    Returns: numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    h_out = ((h + 2 * ph - kh) // sh) + 1
    w_out = ((w + 2 * pw - kw) // sw) + 1

    # Выходной массив теперь 4D: (m, h_out, w_out, nc)
    convolved = np.zeros((m, h_out, w_out, nc))

    # Три разрешенных цикла
    for i in range(h_out):
        for j in range(w_out):
            for k in range(nc):
                # Вырезаем окно из всех картинок и умножаем на k-ое ядро
                # Суммируем по осям высоты, ширины и каналов
                convolved[:, i, j, k] = np.sum(
                    images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] *
                    kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved
