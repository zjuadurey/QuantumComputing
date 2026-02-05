"""
Rollout shadow dynamics using matrix exponential.
"""
import numpy as np
from scipy.linalg import expm


def rollout_shadow(A, o0, t_list):
    """
    Roll out predicted features using matrix exponential.

    o(t) = expm(A * t) @ o(0)

    Args:
        A: Generator matrix of shape (d, d)
        o0: Initial feature vector
        t_list: List of time points

    Returns:
        features_shadow: List of predicted feature vectors
    """
    features_shadow = []

    for t in t_list:
        if t == 0.0:
            features_shadow.append(o0.copy())
        else:
            # Use matrix exponential for stability
            o_t = expm(A * t) @ o0
            features_shadow.append(o_t)

    return features_shadow


def rollout_shadow_stepwise(A, o0, dt, n_steps):
    """
    Roll out predicted features step by step using matrix exponential.

    o_{i+1} = expm(A * dt) @ o_i

    Args:
        A: Generator matrix of shape (d, d)
        o0: Initial feature vector
        dt: Time step
        n_steps: Number of steps

    Returns:
        features_shadow: List of predicted feature vectors (length n_steps + 1)
    """
    features_shadow = [o0.copy()]

    # Precompute propagator
    propagator = expm(A * dt)

    o_current = o0.copy()
    for _ in range(n_steps):
        o_current = propagator @ o_current
        features_shadow.append(o_current.copy())

    return features_shadow
