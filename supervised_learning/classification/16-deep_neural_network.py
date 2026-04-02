#!/usr/bin/env python3
"""Module defining a deep neural network with public attributes."""
import numpy as np


class DeepNeuralNetwork:
    """Class representing a deep neural network."""

    def __init__(self, nx, layers):
        """
        Initializes deep neural network.
        nx: number of input features.
        layers: list of nodes in each layer.
        """
        # 1. Проверка nx (тип, затем значение)
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # 2. Проверка layers (тип и пустота)
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Инициализация публичных атрибутов
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # 3. Цикл инициализации весов (один на всё)
        for i in range(self.L):
            # Проверка элементов списка внутри цикла (чтобы не плодить циклы)
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Размер предыдущего слоя (для первого слоя это nx)
            prev_nodes = nx if i == 0 else layers[i - 1]

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            # Инициализация He et al.
            self.weights[w_key] = np.random.randn(layers[i], prev_nodes) * \
                np.sqrt(2 / prev_nodes)
            # Смещения b инициализируются нулями формы (nodes, 1)
            self.weights[b_key] = np.zeros((layers[i], 1))
