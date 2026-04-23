#!/usr/bin/env python3
"""Module for Isolation Random Tree."""
import numpy as np
Node = __import__('8-build_decision_tree').Node
try:
    Leaf = __import__('8-build_decision_tree').Leaf
except AttributeError:
    Leaf = Node


class Isolation_Random_Tree():
    """Class for Isolation Random Tree for outlier detection."""
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node()
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        return "Isolation_Random_Tree"

    def depth(self):
        """Returns the maximum depth of the tree."""
        def get_depth(node):
            if getattr(node, 'is_leaf', False):
                return node.depth
            return max(get_depth(node.left_child),
                       get_depth(node.right_child))
        return get_depth(self.root)

    def count_nodes(self, only_leaves=False):
        """Counts the number of nodes or leaves in the tree."""
        def count(node):
            if getattr(node, 'is_leaf', False):
                return 1
            if only_leaves:
                return count(node.left_child) + count(node.right_child)
            return 1 + count(node.left_child) + count(node.right_child)
        return count(self.root)

    def update_bounds(self):
        """Dummy implementation for compatibility."""
        pass

    def get_leaves(self):
        """Dummy implementation for compatibility."""
        pass

    def update_predict(self):
        """Updates the prediction function for the tree."""
        def predict_one(x):
            node = self.root
            while not getattr(node, 'is_leaf', False):
                if x[node.feature] > node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            return node.depth
        self.predict = lambda X: np.array([predict_one(x) for x in X])

    def np_extrema(self, arr):
        """Returns the min and max of a numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Selects a random feature and threshold for splitting."""
        feature = self.rng.integers(0, self.explanatory.shape[1])
        feat_vals = self.explanatory[node.sub_population, feature]
        values = np.unique(feat_vals)
        if len(values) <= 1:
            return feature, values[0] if len(values) > 0 else 0
        threshold = self.rng.uniform(values[0], values[-1])
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child node."""
        leaf_child = Node(is_leaf=True, depth=node.depth + 1,
                          sub_population=sub_population)
        leaf_child.value = node.depth + 1
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates a regular child node."""
        child = Node(depth=node.depth + 1, sub_population=sub_population)
        return child

    def fit_node(self, node):
        """Recursively fits a node based on isolation logic."""
        node.feature, node.threshold = self.random_split_criterion(node)

        feat_vals = self.explanatory[node.sub_population, node.feature]
        left_mask = feat_vals > node.threshold
        left_population = node.sub_population[left_mask]
        right_population = node.sub_population[np.logical_not(left_mask)]

        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        len(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         len(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fits the tree on the explanatory data."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.arange(explanatory.shape[0])
        self.root.depth = 0

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
