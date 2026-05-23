#!/usr/bin/env python3
"""Module to perform Expectation-Maximization of a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization of a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    l_prev = 0

    for i in range(iterations):
        g, log_l = expectation(X, pi, m, S)
        if g is None or log_l is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, log_l))

        if i > 0 and abs(log_l - l_prev) <= tol:
            if verbose and (i % 10 != 0):
                print("Log Likelihood after {} iterations: {:.5f}".format(
                    i, log_l))
            return pi, m, S, g, log_l

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        l_prev = log_l

    g, log_l = expectation(X, pi, m, S)
    if g is None or log_l is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after {} iterations: {:.5f}".format(
            iterations, log_l))

    return pi, m, S, g, log_l
