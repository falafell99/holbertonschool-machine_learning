#!/usr/bin/env python3
"""Module to perform pooling back propagation."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer.
    dA: (m, h_new, w_new, c_new) - gradient of cost w.r.t. output.
    A_prev: (m, h_prev, w_prev, c_prev) - previous output.
    kernel_shape: (kh, kw) - size of the pooling kernel.
    stride: (sh, sw) - strides for height and width.
    mode: 'max' or 'avg'.
    Returns: dA_prev.
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    # Границы окна
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'max':
                        # Берем срез из оригинального входа
                        a_prev_slice = A_prev[i, v_start:v_end, h_start:h_end, f]
                        # Создаем маску: 1 там, где был максимум
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # Ошибка уходит только в "максимальный" пиксель
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                            mask * dA[i, h, w, f]
                        )
                    elif mode == 'avg':
                        # Распределяем ошибку равномерно
                        average_grad = dA[i, h, w, f] / (kh * kw)
                        dist = np.ones((kh, kw)) * average_grad
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += dist

    return dA_prev
