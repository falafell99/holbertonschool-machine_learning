#!/usr/bin/env python3
"""Module defining a deep neural network with configurable activations."""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Class representing a deep neural network."""

    def __init__(self, nx, layers, activation='sig'):
        """Initializes deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if i == 0 else layers[i - 1]
            w_key, b_key = "W" + str(i + 1), "b" + str(i + 1)

            self.__weights[w_key] = np.random.randn(layers[i], prev_nodes) * \
                np.sqrt(2 / prev_nodes)
            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for __L."""
        return self.__L

    @property
    def cache(self):
        """Getter for __cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for __weights."""
        return self.__weights

    @property
    def activation(self):
        """Getter for __activation."""
        return self.__activation

    def forward_prop(self, X):
        """Calculates forward propagation with Softmax/Sigmoid/Tanh."""
        self.__cache["A0"] = X
        for i in range(self.__L):
            w_key, b_key = "W" + str(i + 1), "b" + str(i + 1)
            a_prev, a_curr = "A" + str(i), "A" + str(i + 1)

            Z = np.dot(self.__weights[w_key], self.__cache[a_prev]) + \
                self.__weights[b_key]

            if i == self.__L - 1:
                # Multiclass Softmax output
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache[a_curr] = t / np.sum(t, axis=0, keepdims=True)
            else:
                # Hidden layers activation
                if self.__activation == 'sig':
                    self.__cache[a_curr] = 1 / (1 + np.exp(-Z))
                else:
                    self.__cache[a_curr] = np.tanh(Z)

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates categorical cross-entropy cost."""
        m = Y.shape[1]
        # Clean calculation to match checker precision
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        max_indices = np.argmax(A, axis=0)
        prediction = np.eye(A.shape[0])[max_indices].T
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates gradient descent pass."""
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            w_key, b_key = "W" + str(i), "b" + str(i)
            a_prev_key = "A" + str(i - 1)

            W_curr = self.__weights[w_key]
            A_prev = cache[a_prev_key]

            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            if i > 1:
                if self.__activation == 'sig':
                    dz = np.dot(W_curr.T, dz) * (A_prev * (1 - A_prev))
                else:
                    dz = np.dot(W_curr.T, dz) * (1 - (A_prev ** 2))

            self.__weights[w_key] -= alpha * dw
            self.__weights[b_key] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the network with strict logging."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs, it_list = [], []
        for i in range(iterations + 1):
            if i != 0:
                self.forward_prop(X)
                self.gradient_descent(Y, self.__cache, alpha)

            if i % step == 0 or i == iterations:
                A, _ = self.forward_prop(X)
                current_cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i,
                                                                current_cost))
                if graph:
                    costs.append(current_cost)
                    it_list.append(i)

        if graph:
            plt.plot(it_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves object to pickle file."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads object from pickle file."""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
