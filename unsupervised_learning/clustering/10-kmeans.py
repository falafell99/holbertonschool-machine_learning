#!/usr/bin/env python3
"""Module to perform K-means clustering using sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset using sklearn.
    """
    # Создаем объект модели K-means
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)

    # Обучаем модель на наших данных X
    kmeans_model.fit(X)

    # Извлекаем центроиды
    C = kmeans_model.cluster_centers_

    # Извлекаем индексы кластеров для каждой точки
    clss = kmeans_model.labels_

    return C, clss
