#!/usr/bin/env python3
"""Module to update variables using RMSProp optimization algorithm."""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.
    alpha: the learning rate.
    beta2: the RMSProp weight (discounting factor).
    epsilon: a small number to avoid division by zero.
    var: numpy.ndarray containing the variable to be updated.
    grad: numpy.ndarray containing the gradient of var.
    s: the previous second moment of var.
    Returns: the updated variable and the new moment, respectively.
    """
    # Вычисляем новый второй момент (среднее квадратов градиентов)
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    # Обновляем переменную с адаптивным шагом
    var_new = var - alpha * (grad / (np.sqrt(s_new) + epsilon))

    return var_new, s_new
