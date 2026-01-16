#!/usr/bin/env python3
"""Plot a line graph."""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot y = x^3 as a red line from 0 to 10."""
    y = np.arange(0, 11) ** 3
    
    # Create x values from 0 to 10
    x = np.arange(0, 11)
    
    plt.figure(figsize=(6.4, 4.8))
    
    # Plot y as a solid red line
    plt.plot(x, y, 'r-')
    
    # Set x-axis limits
    plt.xlim(0, 10)
    
    # Display the plot
    plt.show()
