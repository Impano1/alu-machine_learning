#!/usr/bin/env python3
"""Calculates the Shannon entropy and P affinities for a data point in t-SNE."""

import numpy as np


def HP(Di, beta):
    """Calculates the Shannon entropy and P affinities for a data point.

    Args:
        Di: numpy.ndarray of shape (n - 1,) containing the pairwise distances
        beta: numpy.ndarray of shape (1,) containing the beta value

    Returns:
        Hi: the Shannon entropy of the points
        Pi: numpy.ndarray of shape (n - 1,) containing the P affinities
    """
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    if sum_Pi == 0:
        sum_Pi = 1e-8
    Pi = Pi / sum_Pi
    Hi = -np.sum(Pi * np.log2(Pi + 1e-10))
    return Hi, Pi
