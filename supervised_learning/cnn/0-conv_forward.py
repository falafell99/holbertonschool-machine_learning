#!/usr/bin/env python3
"""Module to perform convolutional forward propagation."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.
    A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - previous output.
    W: numpy.ndarray (kh, kw, c_prev, c_new) - kernels.
    b: numpy.ndarray (1, 1, 1, c_new) - biases.
    activation: activation function to be applied.
    padding: string, either 'same' or 'valid'.
    stride: tuple (sh, sw).
    Returns: the output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, c_new = W.shape
    sh, sw = stride

    # Определяем паддинг
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Вычисляем выходные размеры
    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    # Добавляем паддинг
    A_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Инициализируем выход
    output = np.zeros((m, h_out, w_out, c_new))

    # Три цикла: по высоте, по ширине и по количеству фильтров на выходе
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                # Слайс: (m, kh, kw, c_prev)
                # Умножаем на k-ое ядро и суммируем по всем осям кроме m
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw

                output[:, i, j, k] = np.sum(
                    A_padded[:, v_start:v_end, h_start:h_end, :] *
                    W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    # Добавляем смещение (bias) и применяем функцию активации
    return activation(output + b)
