#!/usr/bin/env python3
"""
Module for building a decision tree model.
Includes Node, Leaf, and Decision_Tree classes.
"""
import numpy as np


class Node:
    """
    Class that represents an internal node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes a node with feature, threshold, and children.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = {}
        self.upper = {}
        self.indicator = None

    def max_depth_below(self):
        """
        Calculates the maximum depth of the tree starting from this node.
        """
        if self.is_leaf:
            return self.depth
        left_depth = 0
        right_depth = 0
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the total number of nodes or only leaves below this node.
        """
        if only_leaves:
            if self.is_leaf:
                return 1
            count = 0
            if self.left_child:
                count += self.left_child.count_nodes_below(only_leaves=True)
            if self.right_child:
                count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        count = 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves=False)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves=False)
        return count

    def left_child_add_prefix(self, text):
        """
        Adds the necessary prefix for the left child string representation.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds the necessary prefix for the right child string representation.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """
        Returns a string representation of the node and its children.
        """
        if self.is_root:
            out = "root "
        else:
            out = "node "
        out += "[feature={}, threshold={}]\n".format(self.feature,
                                                     self.threshold)
        if self.left_child:
            out += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            out += self.right_child_add_prefix(str(self.right_child))
        return out

    def get_leaves_below(self):
        """
        Retrieves a list of all leaf nodes existing below this node.
        """
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively updates feature bounds for all nodes in the subtree.
        """
        if self.is_root:
            self.lower = {}
            self.upper = {}

        for child, side in [(self.left_child, 'L'), (self.right_child, 'R')]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if side == 'L':
                    # In left child: feature > threshold
                    if self.feature in child.lower:
                        child.lower[self.feature] = max(
                            child.lower[self.feature], self.threshold)
                    else:
                        child.lower[self.feature] = self.threshold
                else:
                    # In right child: feature <= threshold
                    if self.feature in child.upper:
                        child.upper[self.feature] = min(
                            child.upper[self.feature], self.threshold)
                    else:
                        child.upper[self.feature] = self.threshold
                child.update_bounds_below()

    def update_indicator(self):
        """
        Computes the indicator function based on feature bounds.
        """
        def is_large_enough(x):
            """Checks if input satisfies lower bounds."""
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            return np.all([x[:, f] > v for f, v in self.lower.items()], axis=0)

        def is_small_enough(x):
            """Checks if input satisfies upper bounds."""
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            return np.all([x[:, f] <= v for f, v in self.upper.items()], axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))

    def pred(self, x):
        """
        Predicts the class for a single input sample.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """
    Class that represents a leaf node in a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initializes a leaf node with a prediction value.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of this leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since a leaf node is a single node.
        """
        return 1

    def __str__(self):
        """
        Returns the string representation of the leaf node.
        """
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """
        Returns a list containing only this leaf node.
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaf nodes do not have children, so bounds update is empty.
        """
        pass

    def pred(self, x):
        """
        Returns the prediction value of the leaf.
        """
        return self.value


class Decision_Tree:
    """
    Class that represents a complete decision tree model.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes the decision tree with hyperparameters.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth reached in the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Returns the total number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns the string representation of the entire tree.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Returns a list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates feature bounds for the whole tree starting from the root.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Creates a vectorized prediction function for the tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_array(A):
            """Internal function to predict an array of samples."""
            P = np.zeros(A.shape[0], dtype=self.target.dtype)
            for leaf in leaves:
                P[leaf.indicator(A)] = leaf.value
            return P

        self.predict = predict_array

    def pred(self, x):
        """
        Predicts the class for a single input sample.
        """
        return self.root.pred(x)

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum values of a NumPy array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Determines a random feature and threshold for splitting a node.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            f_vals = self.explanatory[:, feature][node.sub_population]
            f_min, f_max = self.np_extrema(f_vals)
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates a leaf child node for a given sub-population.
        """
        targets = self.target[sub_population]
        vals, counts = np.unique(targets, return_counts=True)
        leaf_value = vals[np.argmax(counts)]
        leaf_child = Leaf(leaf_value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates an internal node child for a given sub-population.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively trains a single node by splitting it into children.
        """
        node.feature, node.threshold = self.split_criterion(node)
        l_mask = self.explanatory[:, node.feature] > node.threshold
        r_mask = self.explanatory[:, node.feature] <= node.threshold
        l_pop = np.logical_and(node.sub_population, l_mask)
        r_pop = np.logical_and(node.sub_population, r_mask)

        # Left side
        if (np.sum(l_pop) < self.min_pop or node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[l_pop])) == 1):
            node.left_child = self.get_leaf_child(node, l_pop)
        else:
            node.left_child = self.get_node_child(node, l_pop)
            self.fit_node(node.left_child)

        # Right side
        if (np.sum(r_pop) < self.min_pop or node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[r_pop])) == 1):
            node.right_child = self.get_leaf_child(node, r_pop)
        else:
            node.right_child = self.get_node_child(node, r_pop)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """
        Trains the decision tree on the provided dataset.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            acc = self.accuracy(self.explanatory, self.target)
            print("  Training finished.")
            print("- Depth                     : {}".format(self.depth()))
            print("- Number of nodes           : {}".format(self.count_nodes()))
            print("- Number of leaves          : {}".format(
                self.count_nodes(only_leaves=True)))
            print("- Accuracy on training data : {}".format(acc))

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the model on a test dataset.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def Gini_split_criterion(self, node):
        """
        Placeholder for the Gini split criterion.
        """
        pass
