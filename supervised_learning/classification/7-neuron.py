#!/usr/bin/env python3
"""Module that defines a single neuron with verbose training and graphing."""
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """Class representing a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initializes the neuron.
        nx is the number of input features.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector __W."""
        return self.__W

    @property
    def b(self):
        """Getter for the bias __b."""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output __A."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        Updates and returns the private attribute __A.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Y: correct labels for the input data.
        A: activated output for each example.
        Returns the cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.
        X: input data.
        Y: correct labels.
        Returns predictions and the cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.
        Updates __W and __b.
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.dot(dz, X.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron with options for verbose output and graphing.
        X: input data.
        Y: correct labels.
        iterations: number of iterations.
        alpha: learning rate.
        verbose: boolean to print information.
        graph: boolean to graph information.
        step: step interval for verbose/graph.
        Returns evaluation of training data after iterations.
        """
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

        costs = []
        iters = []

        for i in range(iterations + 1):
            if i != 0:
                self.gradient_descent(X, Y, self.__A, alpha)

            self.forward_prop(X)

            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, self.__A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i,
                                                                current_cost))
                if graph:
                    costs.append(current_cost)
                    iters.append(i)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
