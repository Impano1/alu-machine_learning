#!/usr/bin/env python3
"""
t-SNE module to perform dimensionality reduction on a dataset.
"""

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, perplexity, iterations=1000, dim=2):
    """Performs a t-SNE transformation on dataset X"""
    n, d = X.shape
    alpha = 500.0
    eta = alpha
    momentum = 0.5
    final_momentum = 0.8

    # Reduce dimensionality with PCA first
    X_pca = pca(X, ndim=50)  # PCA preprocessing for t-SNE

    # Compute P affinities
    P = P_affinities(X_pca, perplexity=perplexity)
    P = P + P.T  # Symmetrize P
    P = P / np.sum(P)  # Normalize P
    P = np.maximum(P, 1e-12)  # Avoid numerical instability

    # Initialize low-dimensional Y randomly
    Y = np.random.randn(n, dim)
    dY = np.zeros((n, dim))
    iY = np.zeros((n, dim))  # momentum term

    for i in range(iterations):
        dY = grads(Y, P)  # Compute gradient
        iY = momentum * iY - eta * dY  # momentum update
        Y += iY  # update Y positions

        if i >= 250:
            momentum = final_momentum  # switch to final momentum

        # Print cost every 100 iterations
        if (i + 1) % 100 == 0:
            C = cost(P, Y)
            print("Cost at iteration {}: {}".format(i + 1, C))

    return Y
