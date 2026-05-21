#!/usr/bin/env python3
"""Module to perform a complete t-SNE transformation"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation on a dataset.
    """
    # 1. Предварительное снижение размерности через PCA (ускоряет работу)
    X_pca = pca(X, idims)

    # 2. Вычисление симметричных вероятностей (P) в исходном пространстве
    P = P_affinities(X_pca, perplexity=perplexity)

    # 3. "Раннее преувеличение" (Early Exaggeration)
    # Увеличиваем P в 4 раза, чтобы заставить кластеры сильнее стягиваться
    P = P * 4.0

    # 4. Инициализация переменных
    n = X.shape[0]
    # Начинаем с абсолютно случайного расположения точек на 2D-плоскости
    Y = np.random.randn(n, ndims)
    # Переменная для "инерции" (Momentum) градиентного спуска
    iY = np.zeros((n, ndims))

    # 5. Цикл оптимизации (Градиентный спуск)
    for i in range(1, iterations + 1):
        # Вычисляем градиенты и текущую матрицу Q
        dY, Q = grads(Y, P)

        # Моментум (alpha): 0.5 для первых 20 шагов, затем 0.8
        alpha = 0.5 if i <= 20 else 0.8

        # ИСПРАВЛЕНИЕ ОШИБКИ ИЗ ОРИГИНАЛЬНОЙ СТАТЬИ t-SNE:
        # В статье указан плюс (+) перед градиентом, но для спуска нужен минус (-).
        # Формируем "скорость" с учетом инерции прошлого шага
        iY = alpha * iY - lr * dY

        # Обновляем координаты точек
        Y = Y + iY

        # Центрируем точки вокруг нуля (чтобы они не улетели за пределы экрана)
        Y = Y - np.mean(Y, axis=0)

        # Печатаем стоимость каждые 100 итераций
        if i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

        # Выключаем "раннее преувеличение" после 100 итераций
        if i == 100:
            P = P / 4.0

    return Y
