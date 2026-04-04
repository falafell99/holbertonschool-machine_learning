#!/usr/bin/env python3
"""Module to perform convolutional back propagation."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer.
    dZ: numpy.ndarray (m, h_new, w_new, c_new) - gradient of cost w.r.t. output.
    A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - previous output.
    W: numpy.ndarray (kh, kw, c_prev, c_new) - kernels.
    b: numpy.ndarray (1, 1, 1, c_new) - biases.
    padding: string, 'same' or 'valid'.
    stride: tuple (sh, sw).
    Returns: dA_prev, dW, db.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride

    # Определяем паддинг, который использовался при forward prop
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Добавляем паддинг к входным активациям и создаем пустой массив для dA_prev
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )
    dA_prev_pad = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Основной цикл по примерам, координатам выхода и фильтрам
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    # Границы текущего окна (slice)
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    # Обновляем градиент по весам: dW += a_slice * dZ
                    a_slice = A_prev_pad[i, v_start:v_end, h_start:h_end, :]
                    dW[:, :, :, f] += a_slice * dZ[i, h, w, f]

                    # Обновляем градиент по входу: dA += W * dZ
                    dA_prev_pad[i, v_start:v_end, h_start:h_end, :] += (
                        W[:, :, :, f] * dZ[i, h, w, f]
                    )

    # Убираем паддинг из dA_prev, чтобы вернуть оригинальный размер
    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
