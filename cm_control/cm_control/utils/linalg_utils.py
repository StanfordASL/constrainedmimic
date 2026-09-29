"""Linalg utils"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
import numpy as np


def skew(v: Array) -> Array:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (Array): Vector to convert, shape (3,)

    Returns:
        Array: (3, 3) skew-symmetric matrix
    """
    assert v.shape == (3,)
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def skew_numpy(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (np.ndarray): Vector to convert, shape (3,)

    Returns:
        np.ndarray: (3, 3) skew-symmetric matrix
    """
    assert v.shape == (3,)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


##############
# Experimental
##############


# def right_inv(A: Array) -> Array:
#     # Right inverse of a (wide) matrix that is known to be full-rank
#     return A.T @ jnp.linalg.inv(A @ A.T)


# def damped_right_inv(A: Array, damping: float) -> Array:
#     # Right inverse with added numerical regularization
#     return A.T @ jnp.linalg.inv(A @ A.T + damping**2 * jnp.eye(A.shape[0]))


def _spd_inv(A: Array) -> Array:
    # Inverse of a symmetric positive definite matrix
    return jsp.linalg.solve(A, jnp.eye(A.shape[0]), assume_a="pos")


def _right_inv(A: Array) -> Array:
    # Right inverse of a (wide) matrix that is known to be full-rank
    # Solved with jax.scipy's internal cholesky solver for PD matrices
    return jsp.linalg.solve(A @ A.T, A, assume_a="pos").T


def _damped_right_inv(A: Array, damping: float) -> Array:
    # Right inverse with added numerical regularization
    # Solved with jax.scipy's internal cholesky solver for PD matrices
    return jsp.linalg.solve(
        A @ A.T + damping**2 * jnp.eye(A.shape[0]), A, assume_a="pos"
    ).T
