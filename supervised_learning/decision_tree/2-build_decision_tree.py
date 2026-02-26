#!/usr/bin/env python3
"""Module to build and print a Decision Tree"""
import numpy as np


class Node:
    """Represents an internal node in a decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=None, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def left_child_add_prefix(self, text):
        """Adds prefix for a left child's string representation"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Adds prefix for a right child's string representation"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """Returns the string representation of the node"""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"

        if self.left_child:
            out += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            out += self.right_child_add_prefix(self.right_child.__str__())
        return out[:-1]


class Leaf:
    """Represents a leaf in a decision tree"""
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Returns the string representation of the leaf"""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Represents a decision tree"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="gini", root=None):
        self.root = root
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.split_criterion = split_criterion

    def __str__(self):
        """Returns the string representation of the entire tree"""
        return self.root.__str__()
