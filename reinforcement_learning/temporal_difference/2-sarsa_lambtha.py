#!/usr/bin/env python3
"""
Module for implementing the SARSA(λ) algorithm.
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm.

    Args:
        env: The openAI environment instance.
        Q: A numpy.ndarray of shape (s,a) containing the Q table.
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.
        epsilon: The initial threshold for epsilon greedy.
        min_epsilon: The minimum value that epsilon should decay to.
        epsilon_decay: The decay rate for updating epsilon
                        between episodes.

    Returns:
        Q: The updated Q table.
    """

    for episode in range(episodes):
        E = np.zeros_like(Q)
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state,
                                      next_action] - Q[state, action]
            E[state, action] += 1

            Q += alpha * delta * E
            E *= gamma * lambtha

            state = next_state
            action = next_action

            if done:
                break

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q


def epsilon_greedy(Q, state, epsilon):
    """
    Applies epsilon-greedy policy to select an action.

    Args:
        Q: The Q table.
        state: The current state.
        epsilon: The exploration rate.

    Returns:
        The selected action.
    """

    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])  # Explore
    else:
        return np.argmax(Q[state, :])  # Exploit
