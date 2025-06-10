#!/usr/bin/env python3
"""Initialize variables for t-SNE."""

import numpy as np


def P_init(X, perplexity):
    """Initializes all variables required to calculate the P affinities in t-SNE.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset to be
           transformed
        perplexity: the perplexity that all Gaussian distributions should have

    Returns:
        D: numpy.ndarray of shape (n, n) with the squared pairwise distances
        P: numpy.ndarray of shape (n, n) initialized to all 0’s
        betas: numpy.ndarray of shape (n, 1) initialized to all 1’s
        H: Shannon entropy for perplexity with base 2
    """
    n, d = X.shape

    # Compute pairwise squared Euclidean distances
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # Set diagonal to zero (distance to self)
    np.fill_diagonal(D, 0)

    # Initialize P, betas
    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    # Compute Shannon entropy for the given perplexity
    H = np.log2(perplexity)

    return D, P, betas, H
