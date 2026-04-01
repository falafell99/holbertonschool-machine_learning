#!/usr/bin/env python3
"""Decision tree building blocks."""
import numpy as np


class Node:
    """Represents a node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
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
        """Calculate max depth below this node."""
        left_depth = self.depth
        right_depth = self.depth

        if self.left_child is not None:
            left_depth = self.left_child.max_depth_below()
        if self.right_child is not None:
            right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node."""
        if only_leaves:
            count = 0
            if self.left_child is not None:
                count += self.left_child.count_nodes_below(only_leaves=True)
            if self.right_child is not None:
                count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        else:
            count = 1
            if self.left_child is not None:
                count += self.left_child.count_nodes_below(only_leaves=False)
            if self.right_child is not None:
                count += self.right_child.count_nodes_below(only_leaves=False)
            return count

    def left_child_add_prefix(self, text):
        """Add prefix for left child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return string representation of node."""
        if self.is_root:
            result = "root "
        else:
            result = "node "
        result += f"[feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result

    def get_leaves_below(self):
        """Get all leaves below this node."""
        leaves = []
        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update bounds for this node and its children."""
        if self.is_root:
            self.lower = {}
            self.upper = {}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            if self.feature in self.left_child.upper:
                self.left_child.upper[self.feature] = min(
                    self.left_child.upper[self.feature], self.threshold
                )
            else:
                self.left_child.upper[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            if self.feature in self.right_child.lower:
                self.right_child.lower[self.feature] = max(
                    self.right_child.lower[self.feature], self.threshold
                )
            else:
                self.right_child.lower[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Update indicator function for this node."""
        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            conditions = []
            for key, value in self.lower.items():
                conditions.append(x[:, key] > value)
            return np.all(conditions, axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            conditions = []
            for key, value in self.upper.items():
                conditions.append(x[:, key] <= value)
            return np.all(conditions, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Predict for a single sample."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Represents a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Calculate max depth below this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this leaf."""
        return 1

    def __str__(self):
        """Return string representation of leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Get all leaves below this leaf."""
        return [self]

    def update_bounds_below(self):
        """Update bounds for this leaf."""
        pass

    def update_indicator(self):
        """Update indicator function for this leaf."""
        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            conditions = []
            for key, value in self.lower.items():
                conditions.append(x[:, key] > value)
            return np.all(conditions, axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            conditions = []
            for key, value in self.upper.items():
                conditions.append(x[:, key] <= value)
            return np.all(conditions, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Predict for a single sample."""
        return self.value


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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
        """Calculate depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return string representation of tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Get all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Update prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_array(A):
            n_samples = A.shape[0]
            predictions = np.zeros(n_samples, dtype=int)
            for leaf in leaves:
                indicators = leaf.indicator(A)
                predictions[indicators] = leaf.value
            return predictions

        self.predict = predict_array

    def pred(self, x):
        """Predict for a single sample."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return min and max of array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split criterion."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
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
        """Fit a node recursively."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        left_size = np.sum(left_population)
        right_size = np.sum(right_population)

        left_unique = len(np.unique(self.target[left_population]))
        right_unique = len(np.unique(self.target[right_population]))

        is_left_leaf = (
            left_size < self.min_pop or
            node.depth + 1 >= self.max_depth or
            left_unique == 1
        )
        is_right_leaf = (
            right_size < self.min_pop or
            node.depth + 1 >= self.max_depth or
            right_unique == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree."""
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
            accuracy = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(only_leaves=True)}
- Accuracy on training data : {accuracy}""")

    def accuracy(self, test_explanatory, test_target):
        """Calculate accuracy."""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target
        )) / test_target.size
