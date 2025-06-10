#!/usr/bin/env python3
"""Calculates the gradients for t-SNE."""

import numpy as np


def grads(Y, P):
    """
    Calculates the gradients of Y and the Q affinities.

    Args:
        Y: numpy.ndarray of shape (n, ndim) - low-dimensional embedding
        P: numpy.ndarray of shape (n, n) - high-dimensional affinities (P matrix)

    Returns:
        dY: numpy.ndarray of shape (n, ndim) - gradients of Y (without scalar 4)
        Q: numpy.ndarray of shape (n, n) - low-dimensional affinities
    """
    Q_affinities = __import__('5-Q_affinities').Q_affinities
    Q, num = Q_affinities(Y)
    n, ndim = Y.shape

    # Compute the difference matrix for all pairs (Y_i - Y_j)
    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # shape (n, n, ndim)

    # Compute the pairwise affinities difference (P - Q)
    PQ_diff = P - Q  # shape (n, n)

    # Multiply each pairwise difference by the corresponding PQ_diff * num (student t kernel numerator)
    # As per t-SNE gradient formula, without multiplying by 4 scalar
    # dY_i = sum_j (P_ij - Q_ij) * num_ij * (Y_i - Y_j)
    # num is the numerator of Q affinities = 1 / (1 + dist^2)
    # PQ_diff is (P - Q)
    # We broadcast PQ_diff * num along the third dimension to multiply by (Y_i - Y_j)
    dY = np.einsum('ij,ij,ijk->ik', PQ_diff, num, diff)

    return dY, Q
