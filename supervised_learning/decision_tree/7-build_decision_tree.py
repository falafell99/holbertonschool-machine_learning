#!/usr/bin/env python3
"""Module for building a decision tree."""
import numpy as np


class Node:
    """Class representing a node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a node."""
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
        """Calculate the max depth below this node."""
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
        """Count the number of nodes below."""
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
        """Add prefix for left child string."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child string."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return string representation."""
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
        """Return list of leaves."""
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update bounds for children."""
        if self.is_root:
            self.lower = {}
            self.upper = {}

        for child, direction in [(self.left_child, 'left'),
                                 (self.right_child, 'right')]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if direction == 'left':
                    if self.feature in child.lower:
                        child.lower[self.feature] = max(
                            child.lower[self.feature], self.threshold)
                    else:
                        child.lower[self.feature] = self.threshold
                else:
                    if self.feature in child.upper:
                        child.upper[self.feature] = min(
                            child.upper[self.feature], self.threshold)
                    else:
                        child.upper[self.feature] = self.threshold
                child.update_bounds_below()

    def update_indicator(self):
        """Update indicator function."""
        def is_large_enough(x):
            """Check lower bounds."""
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            conditions = [x[:, f] > val for f, val in self.lower.items()]
            return np.all(conditions, axis=0)

        def is_small_enough(x):
            """Check upper bounds."""
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            conditions = [x[:, f] <= val for f, val in self.upper.items()]
            return np.all(conditions, axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))

    def pred(self, x):
        """Predict for one sample."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Leaf node."""

    def __init__(self, value, depth=None):
        """Init leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Depth of leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count leaf."""
        return 1

    def __str__(self):
        """Str of leaf."""
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """List of this leaf."""
        return [self]

    def update_bounds_below(self):
        """No bounds for leaf."""
        pass

    def pred(self, x):
        """Predict value."""
        return self.value


class Decision_Tree:
    """Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Init tree."""
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
        """Max depth."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Str representation."""
        return self.root.__str__()

    def get_leaves(self):
        """List leaves."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Vectorized predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_array(A):
            """Predict for array."""
            P = np.zeros(A.shape[0], dtype=self.target.dtype)
            for leaf in leaves:
                P[leaf.indicator(A)] = leaf.value
            return P

        self.predict = predict_array

    def pred(self, x):
        """Predict sample."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Min/max."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            f_vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(f_vals)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create leaf child."""
        targets = self.target[sub_population]
        values, counts = np.unique(targets, return_counts=True)
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursive fit."""
        node.feature, node.threshold = self.split_criterion(node)
        l_mask = self.explanatory[:, node.feature] > node.threshold
        r_mask = self.explanatory[:, node.feature] <= node.threshold
        l_pop = np.logical_and(node.sub_population, l_mask)
        r_pop = np.logical_and(node.sub_population, r_mask)

        if (np.sum(l_pop) < self.min_pop or node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[l_pop])) == 1):
            node.left_child = self.get_leaf_child(node, l_pop)
        else:
            node.left_child = self.get_node_child(node, l_pop)
            self.fit_node(node.left_child)

        if (np.sum(r_pop) < self.min_pop or node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[r_pop])) == 1):
            node.right_child = self.get_leaf_child(node, r_pop)
        else:
            node.right_child = self.get_node_child(node, r_pop)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Train tree."""
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
            print(f"""  Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(only_leaves=True)}
- Accuracy on training data : {acc}""")

    def accuracy(self, test_explanatory, test_target):
        """Calculate accuracy."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def Gini_split_criterion(self, node):
        """Gini criterion placeholder."""
        pass
