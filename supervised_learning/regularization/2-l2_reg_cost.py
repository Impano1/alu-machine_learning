#!/usr/bin/env python3
""" Implements l2_reg_cost
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """ Calculates the l2 cost of a tensorflow network.
    Parameters:
    cost (tensorflow.Tensor): a tensor representing the cost
    of the network without the l2 regularisation term.

    Returns:
    tensorflow.Tensor: a tensor representing the cost of the
    network with the l2 regularisation term.
    """
    weights = tf.trainable_weights()

    l2_term = tf.add_n([tf.reduce_sum(tf.square(w)) for w in weights])

    return cost + l2_term
