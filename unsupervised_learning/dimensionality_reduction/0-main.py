#!/usr/bin/env python3
"""
Main file to test t-SNE
"""

import numpy as np
tsne = __import__('8-tsne').tsne

X1 = np.loadtxt('mnist2500_X.txt')
Y1 = tsne(X1, perplexity=30.0, iterations=100)  # perplexity is required here

print(Y1)
