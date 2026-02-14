#!/usr/bin/env python3
"""Module to plot a stacked bar graph"""
import matplotlib.pyplot as plt
import numpy as np


def bars():
    """Plots a stacked bar graph of fruit per person"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    plt.figure(figsize=(6.4, 4.8))

    width = 0.5
    bottom = np.zeros(3)

    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=width, bottom=bottom,
                color=colors[i], label=fruits[i])
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()

    plt.show()
