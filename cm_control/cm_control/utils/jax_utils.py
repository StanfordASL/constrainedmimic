"""Assorted JAX utility functions"""

import jax.numpy as jnp


def is_scalar_float(x) -> bool:
    """Check if a value is a scalar float type"""
    return (
        x is not None
        and jnp.ndim(x) == 0
        and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
    )


def is_scalar_int(x) -> bool:
    """Check if a value is a scalar integer type"""
    return (
        x is not None
        and jnp.ndim(x) == 0
        and jnp.issubdtype(jnp.asarray(x).dtype, jnp.integer)
    )
