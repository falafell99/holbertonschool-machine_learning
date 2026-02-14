# Bayesian Probability

This project explores the concepts of Bayesian probability, including likelihood, priors, posteriors, and marginal probability.

## Learning Objectives
* What is Bayesian Probability?
* What is Bayes' Rule?
* What is Likelihood?
* What is a Prior Probability?
* What is a Posterior Probability?
* What is a Marginal Probability?
* How to calculate each using Python and `numpy`.

## Tasks

### [0. Likelihood](./0-likelihood.py)
A function `def likelihood(x, n, P):` that calculates the likelihood of obtaining data given various hypothetical probabilities of developing severe side effects, assuming a binomial distribution.

* `x` is the number of patients that develop severe side effects
* `n` is the total number of patients observed
* `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities
