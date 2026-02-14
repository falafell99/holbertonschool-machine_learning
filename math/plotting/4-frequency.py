#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plots a histogram of student scores for a project"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Рисуем гистограмму
    # bins=range(0, 101, 10) создает интервалы [0-10, 10-20, ..., 90-100]
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Устанавливаем лимиты и деления оси X
    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))

    # Добавляем подписи и заголовок
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    plt.show()
