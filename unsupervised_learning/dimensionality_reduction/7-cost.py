#!/usr/bin/env python3
"""Calculates the cost of the t-SNE transformation."""

import numpy as np


def cost(P, Q):
    """
    Calculates the cost (KL divergence) of the t-SNE transformation.

    Args:
        P: numpy.ndarray of shape (n, n) - high-dimensional affinities
        Q: numpy.ndarray of shape (n, n) - low-dimensional affinities

    Returns:
        C: float - cost (KL divergence)
    """
    # Avoid division by zero and log(0) by clipping P and Q
    eps = 1e-12
    P_clip = np.maximum(P, eps)
    Q_clip = np.maximum(Q, eps)

    # Compute cost = sum P * log(P / Q)
    C = np.sum(P_clip * np.log(P_clip / Q_clip))

    return C
