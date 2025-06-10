#!/usr/bin/env python3
"""Calculates the symmetric P affinities for t-SNE."""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        tol: float, tolerance for the difference in Shannon entropy.
        perplexity: float, desired perplexity for all Gaussian distributions.

    Returns:
        P: numpy.ndarray of shape (n, n) with symmetric P affinities.
    """
    n, d = X.shape
    D, P, betas, _ = P_init(X, perplexity)

    for i in range(n):
        betamin = None
        betamax = None

        # Distances excluding self
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]

        # Initial entropy and P affinities for current beta
        H, thisP = HP(Di, betas[i])

        # Binary search for beta to get entropy close to log2(perplexity)
        Hdiff = H - np.log2(perplexity)
        tries = 0

        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = betas[i].copy()
                if betamax is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + betamax) / 2
            else:
                betamax = betas[i].copy()
                if betamin is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + betamin) / 2

            H, thisP = HP(Di, betas[i])
            Hdiff = H - np.log2(perplexity)
            tries += 1

        # Set P affinities for this point (excluding self)
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Make P symmetric
    P = (P + P.T) / (2 * n)

    return P
