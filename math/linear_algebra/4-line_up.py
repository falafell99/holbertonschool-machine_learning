#!/usr/bin/env python3
"""Add two arrays element-wise."""


def add_arrays(arr1, arr2):
    """Add two arrays element by element.

    Args:
        arr1: First array of numbers
        arr2: Second array of numbers

    Returns:
        New list with element-wise sum, or None if shapes differ
    """
    # Check if arrays have same length
    if len(arr1) != len(arr2):
        return None

    # Create new list with element-wise sum
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
