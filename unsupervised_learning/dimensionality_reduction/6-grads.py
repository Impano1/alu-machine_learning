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
        dY: numpy.ndarray of shape (n, ndim) - gradients of Y (no scalar 4 multiplier)
        Q: numpy.ndarray of shape (n, n) - low-dimensional affinities
    """
    Q_affinities = __import__('5-Q_affinities').Q_affinities
    Q, num = Q_affinities(Y)
    n, ndim = Y.shape

    # Compute differences between all pairs Y_i - Y_j
    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # shape (n, n, ndim)

    # Compute (P - Q) * numerator of Q affinities
    PQ_diff = P - Q

    # Use einsum to compute gradient for each point
    dY = np.einsum('ij,ij,ijk->ik', PQ_diff, num, diff)

    return dY, Q
