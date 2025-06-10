#!/usr/bin/env python3
import numpy as np
P_init = __import__('5-P_init').P_init
calculate_p_grad = __import__('6-P_grad').P_grad
calculate_cost = __import__('4-cost').cost

def tsne(X, perplexity, iterations=1000, alpha=500.0, dim=2):
    """Performs a t-SNE transformation"""
    n, d = X.shape
    momentum = 0.5
    final_momentum = 0.8
    eta = alpha

    # Compute P affinities
    P = P_init(X, perplexity=perplexity)
    P = P + P.T  # Symmetrize
    P /= np.sum(P)  # Normalize
    P = np.maximum(P, 1e-12)  # Prevent numerical errors

    # Initialize Y randomly
    Y = np.random.randn(n, dim)
    dY = np.zeros((n, dim))
    iY = np.zeros((n, dim))  # momentum term

    for i in range(iterations):
        Q, dQ = calculate_p_grad(Y)
        Q = np.maximum(Q, 1e-12)  # Avoid division by zero

        # Compute gradient
        grad = 4 * (P - Q) * dQ
        dY = np.dot(grad, Y)

        # Update Y using gradient and momentum
        iY = momentum * iY - eta * dY
        Y += iY

        # Adjust momentum
        if i >= 250:
            momentum = final_momentum

        # Compute and print cost every 100 iterations
        if (i + 1) % 100 == 0:
            C = calculate_cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

    return Y
