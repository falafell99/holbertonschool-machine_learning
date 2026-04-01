#!/usr/bin/env python3
"""
Module to build a decision tree model.
Includes Node, Leaf, and Decision_Tree classes.
"""
import numpy as np


class Node:
    """
    Represent an internal node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a node.
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
        Calculate the maximum depth of the tree below this node.
        """
        if self.is_leaf:
            return self.depth
        depths = []
        if self.left_child:
            depths.append(self.left_child.max_depth_below())
        if self.right_child:
            depths.append(self.right_child.max_depth_below())
        return max(depths) if depths else self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes or leaves below this node.
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
        Add the prefix for the left child's string representation.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Add the prefix for the right child's string representation.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """
        Provide a string representation of the node.
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
        Return a list of all leaves below this node.
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
        Compute and update the feature bounds for all children.
        """
        if self.is_root:
            self.lower = {}
            self.upper = {}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            if self.feature in self.left_child.lower:
                self.left_child.lower[self.feature] = max(
                    self.left_child.lower[self.feature], self.threshold)
            else:
                self.left_child.lower[self.feature] = self.threshold
            self.left_child.update_bounds_below()

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            if self.feature in self.right_child.upper:
                self.right_child.upper[self.feature] = min(
                    self.right_child.upper[self.feature], self.threshold)
            else:
                self.right_child.upper[self.feature] = self.threshold
            self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Update the indicator function for the current node.
        """
        def is_large_enough(x):
            """ Check if data exceeds lower bounds. """
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            return np.all([x[:, f] > v for f, v in self.lower.items()], axis=0)

        def is_small_enough(x):
            """ Check if data is within upper bounds. """
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            return np.all([x[:, f] <= v for f, v in self.upper.items()], axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))

    def pred(self, x):
        """
        Predict the class for a single individual.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """
    Represent a leaf node in a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Return 1 as it is a single leaf.
        """
        return 1

    def __str__(self):
        """
        Return the string representation of the leaf.
        """
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """
        Return the leaf itself in a list.
        """
        return [self]

    def update_bounds_below(self):
        """
        Empty method as leaves have no children to update.
        """
        pass

    def pred(self, x):
        """
        Return the predicted value of the leaf.
        """
        return self.value


class Decision_Tree:
    """
    Represent a full decision tree model.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize the decision tree.
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
        Return the depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Return the count of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Return the string representation of the tree.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Return a list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Update feature bounds for the whole tree.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Update the predict function for the tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_array(A):
            """ Predict class for each element in array A. """
            P = np.zeros(A.shape[0], dtype=self.target.dtype)
            for leaf in leaves:
                P[leaf.indicator(A)] = leaf.value
            return P

        self.predict = predict_array

    def pred(self, x):
        """
        Predict for a single individual.
        """
        return self.root.pred(x)

    def np_extrema(self, arr):
        """
        Return the min and max of an array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Choose a random feature and threshold to split a node.
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
        Create a leaf child node.
        """
        targets = self.target[sub_population]
        vals, counts = np.unique(targets, return_counts=True)
        leaf_child = Leaf(vals[np.argmax(counts)])
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create an internal node child.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fit a node by splitting it.
        """
        node.feature, node.threshold = self.split_criterion(node)
        l_mask = self.explanatory[:, node.feature] > node.threshold
        r_mask = self.explanatory[:, node.feature] <= node.threshold
        l_pop = np.logical_and(node.sub_population, l_mask)
        r_pop = np.logical_and(node.sub_population, r_mask)

        # Handle Left
        if (np.sum(l_pop) < self.min_pop or node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[l_pop])) == 1):
            node.left_child = self.get_leaf_child(node, l_pop)
        else:
            node.left_child = self.get_node_child(node, l_pop)
            self.fit_node(node.left_child)

        # Handle Right
        if (np.sum(r_pop) < self.min_pop or node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[r_pop])) == 1):
            node.right_child = self.get_leaf_child(node, r_pop)
        else:
            node.right_child = self.get_node_child(node, r_pop)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """
        Train the decision tree model.
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
        Calculate the accuracy of the tree on test data.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def Gini_split_criterion(self, node):
        """
        Placeholder for Gini split criterion.
        """
        pass
