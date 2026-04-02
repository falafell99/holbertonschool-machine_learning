#!/usr/bin/env python3
"""Module to calculate weighted moving average with bias correction."""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.
    data: list of data to calculate the moving average of.
    beta: the weight used for the moving average.
    Returns: a list containing the moving averages of data.
    """
    v = 0
    m_avg = []
    for t in range(1, len(data) + 1):
        # θ_t - текущее значение из данных
        theta_t = data[t - 1]
        # v_t = β * v_{t-1} + (1 - β) * θ_t
        v = beta * v + (1 - beta) * theta_t
        # Применяем Bias Correction: v / (1 - β^t)
        v_corrected = v / (1 - (beta ** t))
        m_avg.append(v_corrected)
    return m_avg
