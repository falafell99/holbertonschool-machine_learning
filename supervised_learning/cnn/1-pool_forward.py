#!/usr/bin/env python3
"""Module to perform pooling forward propagation."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.
    A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - previous output.
    kernel_shape: tuple (kh, kw) - size of the pooling kernel.
    stride: tuple (sh, sw) - strides for height and width.
    mode: string, either 'max' or 'avg'.
    Returns: the output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Вычисляем выходные размеры (в пулинге паддинг обычно не используется)
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # Инициализируем выходной массив (каналы c_prev сохраняются)
    output = np.zeros((m, h_out, w_out, c_prev))

    # Проходим по сетке выходного изображения
    for i in range(h_out):
        for j in range(w_out):
            # Определяем границы окна
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # Извлекаем слайс из всех примеров и всех каналов сразу
            # Слайс имеет форму (m, kh, kw, c_prev)
            A_slice = A_prev[:, v_start:v_end, h_start:h_end, :]

            if mode == 'max':
                # Находим максимум в окне для каждого канала
                output[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                # Считаем среднее значение
                output[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return output
