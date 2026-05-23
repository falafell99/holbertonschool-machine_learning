#!/usr/bin/env python3
"""Module to calculate a GMM using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset using scikit-learn.
    """
    # 1. Создаем объект модели GaussianMixture
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    # 2. Обучаем модель (запускаем EM-алгоритм под капотом)
    gmm_model.fit(X)

    # 3. Извлекаем обученные параметры
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_

    # 4. Получаем предсказания кластеров для каждой точки
    clss = gmm_model.predict(X)

    # 5. Вычисляем BIC
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
