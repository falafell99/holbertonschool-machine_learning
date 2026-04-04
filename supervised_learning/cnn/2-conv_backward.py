#!/usr/bin/env python3
"""Module to perform convolutional back propagation."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer.
    dZ: (m, h_new, w_new, c_new) - gradient of cost w.r.t. output.
    A_prev: (m, h_prev, w_prev, c_prev) - previous output.
    W: (kh, kw, c_prev, c_new) - kernels.
    b: (1, 1, 1, c_new) - biases.
    Returns: dA_prev, dW, db.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    dA_prev_pad = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    v_start, h_start = h * sh, w * sw
                    v_end, h_end = v_start + kh, h_start + kw

                    a_slice = A_prev_pad[i, v_start:v_end, h_start:h_end, :]
                    dW[:, :, :, f] += a_slice * dZ[i, h, w, f]
                    dA_prev_pad[i, v_start:v_end, h_start:h_end, :] += (
                        W[:, :, :, f] * dZ[i, h, w, f]
                    )

    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
