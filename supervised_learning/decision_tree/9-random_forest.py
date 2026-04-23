#!/usr/bin/env python3
"""Module for a Random Forest classifier."""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Class representing a Random Forest classifier."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the Random Forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the class for given data using a majority vote
        from the forest.
        """
        # Generate predictions for each tree in the forest
        # Shape: (n_trees, n_samples)
        predictions = np.array([predict_func(explanatory)
                                for predict_func in self.numpy_preds])

        # Calculate the mode (most frequent) prediction for each example
        # SciPy's mode is efficient, but we can do it purely with NumPy
        # by iterating over columns (samples) or using advanced indexing.
        
        # NumPy-only mode calculation across axis 0 (trees)
        def get_mode(col):
            values, counts = np.unique(col, return_counts=True)
            return values[np.argmax(counts)]

        # Apply get_mode to every column (every sample)
        final_predictions = np.apply_along_axis(get_mode, axis=0,
                                                arr=predictions)

        return final_predictions

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Fits the Random Forest to the training data."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            # Use random splitting as specified by the task description
            # for the trees inside the random forest
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            mean_depth = np.array(depths).mean()
            mean_nodes = np.array(nodes).mean()
            mean_leaves = np.array(leaves).mean()
            mean_acc = np.array(accuracies).mean()
            forest_acc = self.accuracy(self.explanatory, self.target)
            
            print(f"  Training finished.")
            print(f"    - Mean depth                     : {mean_depth}")
            print(f"    - Mean number of nodes           : {mean_nodes}")
            print(f"    - Mean number of leaves          : {mean_leaves}")
            print(f"    - Mean accuracy on training data : {mean_acc}")
            print(f"    - Accuracy of the forest on td   : {forest_acc}")

    def accuracy(self, test_explanatory, test_target):
        """Calculates the accuracy of the Random Forest."""
        predictions = self.predict(test_explanatory)
        return np.sum(np.equal(predictions, test_target)) / test_target.size
