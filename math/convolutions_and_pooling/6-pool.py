#!/usr/bin/env python3
"""Module to perform pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.
    images: numpy.ndarray (m, h, w, c) containing multiple images.
    kernel_shape: tuple (kh, kw) containing the kernel shape.
    stride: tuple (sh, sw).
    mode: 'max' or 'avg'.
    Returns: numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Вычисляем выходные размеры
    h_out = ((h - kh) // sh) + 1
    w_out = ((w - kw) // sw) + 1

    # Инициализируем выходной массив
    pooled = np.zeros((m, h_out, w_out, c))

    # Всего 2 разрешенных цикла по сетке выхода
    for i in range(h_out):
        for j in range(w_out):
            # Вырезаем "кубик" из всех изображений и каналов сразу
            image_slice = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            if mode == 'max':
                # Выбираем максимум по осям высоты и ширины окна (1, 2)
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                # Считаем среднее
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
