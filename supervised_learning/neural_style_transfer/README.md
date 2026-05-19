# Neural Style Transfer (NST)

This project implements the Neural Style Transfer algorithm from scratch using TensorFlow and a pre-trained VGG19 Convolutional Neural Network. NST optimizes a generated image to minimize content loss with a base image and style loss with a reference artwork.

## Project Structure and Pipeline

* `0-neural_style.py`: Core `NST` class initialization. 
  - Defines the specific VGG19 activation layers used for style extraction (`block1_conv1` through `block5_conv1`) and content extraction (`block5_conv2`).
  - Implements defensive programming for robust input validation.
  - Contains `scale_image`, a static method that resizes images to a maximum dimension of 512 pixels using bicubic interpolation and normalizes pixel values to the `[0, 1]` range to prevent Out-Of-Memory (OOM) errors and exploding gradients during the optimization process.
