#!/usr/bin/env python3
"""Module for Early Stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.
    cost: the current validation cost of the neural network.
    opt_cost: the lowest recorded validation cost of the NN.
    threshold: the threshold used for early stopping.
    patience: the patience count used for early stopping.
    count: the count of how long the threshold has not been met.
    Returns: a boolean of whether the network should be stopped early,
             followed by the updated count.
    """
    # Если разница между лучшей ценой и текущей больше порога — прогресс есть
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        # Прогресса нет (или он слишком мал), инкрементируем счетчик терпения
        count += 1

    # Если счетчик дошел до лимита терпения (patience) — пора останавливаться
    if count >= patience:
        return True, count
    return False, count
