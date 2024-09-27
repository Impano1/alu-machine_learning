#!/usr/bin/env python3
""" Implements l2_reg_gradient_descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lam, L):
    """ Updates the weights and biases of a neural network using
    gradients descent and l2 regularization. Weights and biases
    will be updated in place. Assumes tanh activation for each layer
    and softmax for the last layer.

    Parameters:
    Y (numpy.ndarray): A one-hot encoded numpy array with shape 
    (n_classes, m) that contains the correct labels for the data.
    In this context, n_classes is the number of classes and m is the
    number of training examples.

    weights (dict): Is a dictionary storing the weights of the neural
    network. It includes keys such as W1, W2 and b2 for the weights of
    layer 1, layer 2 and the bias of layer 2 respectively. It's values
    are numpy arrays storing the actual weights.

    cache (dict): Is a dictionary storing the outputs of each layer of
    the neural network. It contains keys such as A1, A2 for the
    activations of layer 1 and layer 2 respectively. It contains values
    as numpy arrays.

    alpha (float): The learning rate of the gradient descent algorithm.

    lam (float): The l2 regularization parameter.

    L (int): The number of layers in the neural network.

    Returns:
    None.
    """
