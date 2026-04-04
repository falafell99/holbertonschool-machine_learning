#!/usr/bin/env python3
"""Module for building and printing a Decision Tree."""
import numpy as np


class Decision_Tree:
    """Class representing a Decision Tree."""
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        """Returns the string representation of the tree."""
        return self.root.__str__()


class Leaf:
    """Class representing a leaf in a Decision Tree."""
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth

    def __str__(self):
        """Returns the string representation of a leaf."""
        return (f"-> leaf [value={self.value}]")


class Node:
    """Class representing a node in a Decision Tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=None, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root

    def left_child_add_prefix(self, text):
        """Adds prefixes to the left child string."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds prefixes to the right child string."""
        lines = text.split("\n")
        # Для правой ветви на последующих строках рисуем пробелы вместо '|'
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Recursively builds the string representation of the node."""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]\n"

        # Добавляем левого и правого ребенка с соответствующими префиксами
        out += self.left_child_add_prefix(self.left_child.__str__())
        out += self.right_child_add_prefix(self.right_child.__str__())
        return out
