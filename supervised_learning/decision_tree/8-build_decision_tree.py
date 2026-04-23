#!/usr/bin/env python3
"""Module for building a decision tree with Gini splitting criterion."""
import numpy as np


class Node:
    """Class representing a node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_leaf=False, value=None, depth=0,
                 sub_population=None):
        """Initialization of a node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.value = value
        self.depth = depth
        self.sub_population = sub_population


class Decision_Tree:
    """Class for a Decision Tree classifier."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random"):
        """Initialization of the decision tree."""
        self.rng = np.random.default_rng(seed)
        self.root = None
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the tree."""
        def get_depth(node):
            if node.is_leaf:
                return node.depth
            return max(get_depth(node.left_child), get_depth(node.right_child))
        return get_depth(self.root)

    def count_nodes(self, only_leaves=False):
        """Counts the number of nodes or leaves in the tree."""
        def count(node):
            if node.is_leaf:
                return 1
            if only_leaves:
                return count(node.left_child) + count(node.right_child)
            return 1 + count(node.left_child) + count(node.right_child)
        return count(self.root)

    def fit(self, explanatory, target, verbose=0):
        """Fits the model to the training data."""
        self.explanatory = explanatory
        self.target = target
        self.root = Node(depth=0, sub_population=np.arange(target.size))
        if self.split_criterion == "random":
            self.split_criterion_func = self.random_split_criterion
        elif self.split_criterion == "Gini":
            self.split_criterion_func = self.Gini_split_criterion
        self.grow_tree(self.root)
        self.update_predict()
        if verbose:
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : {self.count_nodes(True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, explanatory, target):
        """Calculates the accuracy of the model on a given dataset."""
        return np.sum(self.predict(explanatory) == target) / target.size

    def grow_tree(self, node):
        """Recursively grows the decision tree."""
        if (node.depth >= self.max_depth or
                len(node.sub_population) <= self.min_pop or
                len(np.unique(self.target[node.sub_population])) == 1):
            node.is_leaf = True
            node.value = np.argmax(np.bincount(self.target[node.sub_population]))
            return

        feature, threshold = self.split_criterion_func(node)
        if feature is None:
            node.is_leaf = True
            node.value = np.argmax(np.bincount(self.target[node.sub_population]))
            return

        node.feature = feature
        node.threshold = threshold

        left_mask = (self.explanatory[:, feature])[node.sub_population] > threshold
        left_sub = node.sub_population[left_mask]
        right_sub = node.sub_population[np.logical_not(left_mask)]

        if len(left_sub) == 0 or len(right_sub) == 0:
            node.is_leaf = True
            node.value = np.argmax(np.bincount(self.target[node.sub_population]))
            return

        node.left_child = Node(depth=node.depth + 1, sub_population=left_sub)
        node.right_child = Node(depth=node.depth + 1, sub_population=right_sub)

        self.grow_tree(node.left_child)
        self.grow_tree(node.right_child)

    def random_split_criterion(self, node):
        """Selects a random feature and threshold for splitting."""
        feature = self.rng.integers(0, self.explanatory.shape[1])
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        if len(values) <= 1:
            return None, None
        threshold = self.rng.uniform(values[0], values[-1])
        return feature, threshold

    def update_predict(self):
        """Updates the prediction function for the tree."""
        def predict_one(x):
            node = self.root
            while not node.is_leaf:
                if x[node.feature] > node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            return node.value
        self.predict = lambda X: np.array([predict_one(x) for x in X])

    def possible_thresholds(self, node, feature):
        """Calculates possible split thresholds for a feature."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Calculates the best Gini impurity split for a single feature."""
        sub_pop = node.sub_population
        X_feat = self.explanatory[sub_pop, feature]
        Y_node = self.target[sub_pop]

        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return np.array([0, 1.0])

        # Представление классов через One-Hot (n, c)
        classes = np.unique(self.target)
        one_hot = (Y_node[:, None] == classes[None, :])

        # Сравнение значений с порогами (n, t)
        gt = X_feat[:, None] > thresholds[None, :]

        # Тензор Left_F формы (n, t, c)
        Left_F = gt[:, :, None] & one_hot[:, None, :]

        # Статистика для левого потомка
        n_L_k = np.sum(Left_F, axis=0)
        n_L = np.sum(n_L_k, axis=1)

        # Статистика для правого потомка
        total_class_counts = np.sum(one_hot, axis=0)
        n_R_k = total_class_counts[None, :] - n_L_k
        n_total = X_feat.shape[0]
        n_R = n_total - n_L

        # Расчет Gini с игнорированием деления на ноль (когда потомок пуст)
        with np.errstate(divide='ignore', invalid='ignore'):
            sum_pk2_L = np.sum((n_L_k / n_L[:, None])**2, axis=1)
            sum_pk2_R = np.sum((n_R_k / n_R[:, None])**2, axis=1)

            gini_L = np.nan_to_num(1 - sum_pk2_L)
            gini_R = np.nan_to_num(1 - sum_pk2_R)

        # Усредненный Gini
        gini_split = (n_L / n_total) * gini_L + (n_R / n_total) * gini_R

        best_idx = np.argmin(gini_split)
        return np.array([thresholds[best_idx], gini_split[best_idx]])

    def Gini_split_criterion(self, node):
        """Chooses the best feature and threshold based on Gini split."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return int(i), X[i, 0]
