#!/usr/bin/env python3
"""Performs a t-SNE transformation."""

import numpy as np


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        ndims: new dimensionality after transformation
        idims: intermediate dimensions for PCA
        perplexity: the perplexity for P affinities
        iterations: total number of iterations to perform
        lr: learning rate

    Returns:
        Y: numpy.ndarray of shape (n, ndims) containing the low dimensional transformation
    """
    pca = __import__('1-pca').pca
    P_affinities = __import__('4-P_affinities').P_affinities
    grads = __import__('6-grads').grads
    cost = __import__('7-cost').cost

    # Step 1: PCA to reduce to idims
    X = pca(X, idims)

    # Step 2: Compute P affinities
    P = P_affinities(X, perplexity=perplexity)
    P = P * 4  # Early exaggeration
    n = X.shape[0]

    # Step 3: Initialize Y randomly
    Y = np.random.randn(n, ndims)

    # Initialize variables for momentum and updates
    dY = 0
    iY = 0
    gains = np.ones((n, ndims))

    for i in range(iterations + 1):
        # Compute gradients and Q
        dY, Q = grads(Y, P)

        # Update gains
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + \
                (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains = np.maximum(gains, 0.01)

        # Momentum schedule
        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8

        # Gradient update
        iY = momentum * iY - lr * (gains * dY)
        Y = Y + iY

        # Re-center Y
        Y = Y - np.mean(Y, axis=0)

        # After 100 iterations, stop early exaggeration
        if i == 100:
            P = P / 4

        # Print cost every 100 iterations (excluding 0)
        if i != 0 and i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

    return Y
