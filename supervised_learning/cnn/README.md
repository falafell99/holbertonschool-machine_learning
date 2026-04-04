# Convolutional Neural Networks

This directory contains implementations of fundamental components of Convolutional Neural Networks (CNNs).

## Tasks
| Task | File | Description |
| **1. Pooling Forward Prop** | `1-pool_forward.py` | Performs forward propagation over a pooling layer (max/avg). |
| --- | --- | --- |
| **0. Convolutional Forward Prop** | `0-conv_forward.py` | Performs forward propagation over a convolutional layer. |

## Formulas
Output dimensions with stride ($s$) and padding ($p$):
$$n_H = \lfloor \frac{n_{Hprev} + 2p_H - f_H}{s_H} \rfloor + 1$$
$$n_W = \lfloor \frac{n_{Wprev} + 2p_W - f_W}{s_W} \rfloor + 1$$
