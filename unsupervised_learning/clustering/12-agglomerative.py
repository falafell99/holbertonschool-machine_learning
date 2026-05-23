#!/usr/bin/env python3
"""Module to perform agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage.
    """
    # 1. Выполняем иерархическую кластеризацию методом Уорда
    # Матрица Z (linkage matrix) содержит всю историю слияний кластеров
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    # 2. Строим дендрограмму
    # color_threshold раскрасит ветви дерева в разные цвета в зависимости
    # от заданного нами порога отсечения (dist)
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)

    # Отображаем график, как того требует задание
    plt.show()

    # 3. Формируем плоские кластеры (срезаем дерево на высоте dist)
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss
