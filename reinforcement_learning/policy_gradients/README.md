# Policy Gradients

This project implements the building blocks of the policy gradient
method for reinforcement learning with `numpy`.

## Tasks

### 0. Simple Policy function

`policy_gradient.py` contains the function `policy(matrix, weight)`
that computes the policy with a weight of a matrix.

* `matrix` is a `numpy.ndarray` of shape `(n, s)` containing the states
* `weight` is a `numpy.ndarray` of shape `(s, a)` containing the weights
* Returns a `numpy.ndarray` of shape `(n, a)` with the probability of
  each action (softmax of `matrix.dot(weight)`)

## Requirements

* Python 3.9
* numpy
