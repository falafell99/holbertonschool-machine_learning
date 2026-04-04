#!/usr/bin/env python3
"""Module to calculate L2 regularization cost in Keras."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    cost: a tensor containing the cost without L2 regularization.
    model: a Keras model that includes layers with L2 regularization.
    Returns: a tensor with the total cost for each layer,
             accounting for L2 regularization.
    """
    # model.losses содержит список тензоров регуляризации для каждого слоя
    # Мы прибавляем базовый cost к каждой из этих потерь
    return cost + model.losses
