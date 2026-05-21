#!/usr/bin/env python3
"""Module to calculate Shannon entropy and P affinities in t-SNE"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    """
    # ТРЮК СТАБИЛЬНОСТИ (Shift Trick):
    # Чтобы np.exp() не выдавал нули при огромных расстояниях,
    # мы сдвигаем все значения так, чтобы максимальное было равно 0.
    # np.exp(0) = 1.0, поэтому у нас всегда будет хотя бы одна единица.
    # Математически дроби (вероятности) от этого вообще не меняются!
    shift = -Di * beta
    shift = shift - np.max(shift)

    Pi_num = np.exp(shift)
    sum_Pi = np.sum(Pi_num)

    # Теперь sum_Pi гарантированно >= 1.0. Деления на ноль не будет.
    Pi = Pi_num / sum_Pi

    # 1e-300 оказалось слишком малым. 1e-15 - это абсолютно безопасный
    # минимум даже для float32, который спасает логарифм от падения в -inf.
    Hi = -np.sum(Pi * np.log2(np.maximum(Pi, 1e-15)))

    return Hi, Pi
