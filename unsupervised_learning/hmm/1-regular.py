#!/usr/bin/env python3

"""
This module determines steady state probabilities
of a markov chain"""

import numpy as np


def regular(P):
    """
    determines steady state probabilities
    of a markov chain

    P - square 2D numpy.ndarray: (n, n) -transition matrix
        - P[i, j] - probability of transitioning from
    state i to state j
        - n no. of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) representing
    steady state probabilities, or None on failure
    """
    try:
        if len(P.shape) != 2:
            return None
        n = P.shape[0]
        if n != P.shape[1]:
            return None

        #  (πP).T = π.T ⟹ P.T π.T = π.T (.)
        evals, evecs = np.linalg.eig(P.T)

        # trick: has to be normalized
        state = (evecs / evecs.sum())

        # P.T π.T = π.T (.)
        new_state = np.dot(state.T, P)
        for i in new_state:
            if (i >= 0).all() and np.isclose(i.sum(), 1):
                return i.reshape(1, n)
    except Exception:
        return None
