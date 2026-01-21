#!/usr/bin/env python3
"""Plot stacked bar graph of fruit quantities."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot stacked bar graph showing fruit per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    plt.figure(figsize=(6.4, 4.8))

    # Define colors for each fruit
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    person_names = ['Farrah', 'Fred', 'Felicia']

    # Initialize bottom for stacking
    bottom = np.zeros(3)

    # Plot each fruit type as stacked bar
    for i in range(4):
        plt.bar(person_names, fruit[i], width=0.5,
                color=colors[i], label=fruit_names[i],
                bottom=bottom)
        bottom += fruit[i]  # Update bottom for next fruit

    # Set labels and title
    plt.ylabel('Quantity of Fruit', fontsize='medium')
    plt.title('Number of Fruit per Person', fontsize='medium')

    # Set y-axis limits and ticks
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Add legend
    plt.legend()

    plt.show()
