#!/usr/bin/env python3
"""
Module for implementing the TD(λ) algorithm.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm.

    Args:
        env: The openAI environment instance.
        V: A numpy.ndarray of shape (s,) containing the value estimate.
        policy: A function that takes in a state and returns
                the next action to take.
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate.
    """

    for _ in range(episodes):
        state = env.reset()
        E = np.zeros_like(V)  # Initialize eligibility trace

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # Update eligibility trace
            E *= lambtha * gamma
            E[state] += 1

            # Update value estimate
            td_error = reward + gamma * V[next_state] - V[state]
            V += alpha * td_error * E

            state = next_state
            if done:
                break

    return V
