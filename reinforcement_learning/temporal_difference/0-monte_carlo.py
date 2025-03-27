#!/usr/bin/env python3
"""
Module for implementing the Monte Carlo algorithm.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: The openAI environment instance.
        V: A numpy.ndarray of shape (s,) containing the value estimate.
        policy: A function that takes in a state and returns
                the next action to take.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate.
    """

    for _ in range(episodes):
        episode = []
        state = env.reset()

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if done:
                break

        G = 0
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])

    return V
