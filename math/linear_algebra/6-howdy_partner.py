#!/usr/bin/env python3
"""Concatenate two arrays."""


def cat_arrays(arr1, arr2):
    """Concatenate two arrays.

    Args:
        arr1: First array
        arr2: Second array

    Returns:
        New list containing all elements from arr1 followed by arr2
    """
    # Create new list and extend with both arrays
    result = []
    result.extend(arr1)
    result.extend(arr2)
    return result
