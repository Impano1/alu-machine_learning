lculates the Shannon entropy and P affinities for a data point in t-SNE."""

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
    # Compute the P affinities using the Gaussian kernel
    Pi = np.exp(-Di * beta)
    
    # Prevent division by zero
    sum_Pi = np.sum(Pi)
    if sum_Pi == 0:
        sum_Pi = 1e-8

    Pi = Pi / sum_Pi

    # Compute Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi + 1e-10))  # 1e-10 for numerical stability

    return Hi, Pi
