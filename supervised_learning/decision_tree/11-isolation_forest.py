#!/usr/bin/env python3
"""Module for Isolation Random Forest for outlier detection."""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Class representing an Isolation Random Forest."""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the Isolation Random Forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Returns the mean depth of the leaves the examples fall into."""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Fits the forest to the explanatory data."""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))

        if verbose == 1:
            mean_depth = np.array(depths).mean()
            mean_nodes = np.array(nodes).mean()
            mean_leaves = np.array(leaves).mean()
            print(f"  Training finished.")
            print(f"    - Mean depth                     : {mean_depth}")
            print(f"    - Mean number of nodes           : {mean_nodes}")
            print(f"    - Mean number of leaves          : {mean_leaves}")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory that have
        the smallest mean depth, along with their depths.
        """
        depths = self.predict(explanatory)

        # Get the indices of the sorted depths in ascending order
        sorted_indices = np.argsort(depths)

        # Select the top 'n_suspects' indices (the smallest depths)
        suspect_indices = sorted_indices[:n_suspects]

        # Extract the corresponding explanatory rows and their depths
        top_suspects = explanatory[suspect_indices]
        top_depths = depths[suspect_indices]

        return top_suspects, top_depths
