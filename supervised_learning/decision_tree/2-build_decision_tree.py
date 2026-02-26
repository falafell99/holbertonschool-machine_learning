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

    def max_depth_below(self):
        """Calculate max depth below this node."""
        if self.left_child is None and self.right_child is None:
            return self.depth

        left_depth = self.depth
        right_depth = self.depth

        if self.left_child is not None:
            left_depth = self.left_child.max_depth_below()
        if self.right_child is not None:
            right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node.

        Args:
            only_leaves: If True, count only leaves

        Returns:
            Number of nodes/leaves below this node
        """
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
            node_str = "root "
        else:
            node_str = "node "
        result = f"{node_str}[feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result


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
        """Count nodes below this leaf.

        Args:
            only_leaves: If True, count only leaves

        Returns:
            Number of nodes/leaves below this leaf (always 1 for leaf)
        """
        return 1

    def __str__(self):
        """Return string representation of leaf."""
        return f"-> leaf [value={self.value}]"


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
        """Count nodes in the tree.

        Args:
            only_leaves: If True, count only leaves

        Returns:
            Number of nodes/leaves in the tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return string representation of tree."""
        return self.root.__str__()
