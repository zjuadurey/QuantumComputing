"""
Fit linear generator matrix A for shadow dynamics.
"""
import numpy as np


def fit_generator(features_list, dt, ridge_lambda=1e-6):
    """
    Fit a real matrix A by least squares with ridge regularization.

    Model: (o_{i+1} - o_i) / dt ≈ A @ o_i

    This is equivalent to: o_dot ≈ A @ o

    Args:
        features_list: List of feature vectors [o_0, o_1, ..., o_T]
        dt: Time step between consecutive features
        ridge_lambda: Ridge regularization parameter

    Returns:
        A: Generator matrix of shape (d, d) where d is feature dimension
    """
    n_times = len(features_list)
    d = len(features_list[0])

    # Build matrices for least squares: Y = A @ X
    # where X = [o_0, o_1, ..., o_{T-1}] and Y = [(o_1-o_0)/dt, ..., (o_T-o_{T-1})/dt]

    X = np.zeros((d, n_times - 1))
    Y = np.zeros((d, n_times - 1))

    for i in range(n_times - 1):
        X[:, i] = features_list[i]
        Y[:, i] = (features_list[i+1] - features_list[i]) / dt

    # Ridge regression: A = Y @ X^T @ (X @ X^T + lambda * I)^{-1}
    XXT = X @ X.T
    reg = ridge_lambda * np.eye(d)
    A = Y @ X.T @ np.linalg.inv(XXT + reg)

    return A
