#!/usr/bin/env python3
"""Calculates the Q affinities for t-SNE."""

import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities and numerator matrix for t-SNE.

    Args:
        Y: numpy.ndarray of shape (n, ndim), low dimensional representation

    Returns:
        Q: numpy.ndarray of shape (n, n) with Q affinities
        num: numpy.ndarray of shape (n, n) with numerator of Q affinities
    """
    n = Y.shape[0]

    # Compute squared Euclidean distance matrix between points
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    # Compute numerator matrix using Student t-distribution kernel (1 + D)^-1
    num = 1 / (1 + D)

    # Set diagonal to zero to ignore self-affinity
    np.fill_diagonal(num, 0)

    # Normalize to get Q affinities
    Q = num / np.sum(num)

    return Q, num
