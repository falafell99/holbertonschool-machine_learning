# Convolutions and Pooling

This project covers the fundamental mathematical operations behind Convolutional Neural Networks (CNNs).

## Tasks
| Task | File | Description |
| **3. Strided Convolution** | `3-convolve_grayscale.py` | Performs convolution with custom padding and strides. |
| **2. Custom Padding** | `2-convolve_grayscale_padding.py` | Performs convolution with user-defined padding (ph, pw). |
| **1. Same Convolution** | `1-convolve_grayscale_same.py` | Performs a same convolution (padding) keeping input dimensions. |
| --- | --- | --- |
| **0. Valid Convolution** | `0-convolve_grayscale_valid.py` | Performs a valid convolution on grayscale images using only 2 loops. |

## Formulas
For an image $(H, W)$ and kernel $(K_h, K_w)$, the output size is:
$$H_{out} = H - K_h + 1$$
$$W_{out} = W - K_w + 1$$
