"""Utilities for computing the support polygon

Note: Some of these functions (particularly the support points)
should probably get integrated into frax

Also, the convex hull logic is likely "good enough" for this
application, but could be improved
"""

from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from frax import Humanoid

from cm_control.utils.transform_utils import transform_points


# @partial(jax.jit, static_argnames=("robot",))
def left_foot_support_points(robot: Humanoid, q: Array) -> Array:
    transform = robot.left_foot_transform(q)
    # NOTE: These seem to be ever so slightly off from the points in mujoco
    # But, these seem to match the values in the URDF?
    x = (0.12 + 0.05) / 2
    y = (0.025 + 0.03) / 2
    z = 0.0
    offsets = jnp.array(
        [
            [x, y, z],
            [-x, y, z],
            [-x, -y, z],
            [x, -y, z],
        ]
    )
    return transform_points(transform, offsets)


# @partial(jax.jit, static_argnames=("robot",))
def right_foot_support_points(robot: Humanoid, q: Array) -> Array:
    transform = robot.right_foot_transform(q)
    # NOTE: These seem to be ever so slightly off from the points in mujoco
    # But, these seem to match the values in the URDF?
    x = (0.12 + 0.05) / 2
    y = (0.025 + 0.03) / 2
    z = 0.0
    offsets = jnp.array(
        [
            [x, y, z],
            [-x, y, z],
            [-x, -y, z],
            [x, -y, z],
        ]
    )
    return transform_points(transform, offsets)


# TODO: This function could likely be improved
# Look into Andrew monotonic chain method? Instead of graham scan
def convex_hull_2d(pts: Array) -> Tuple[Array, int]:
    """Compute the convex hull of a set of points in 2D

    Args:
        pts (Array): Input points to compute the hull of, shape (n, 2)

    Returns:
        Tuple[Array, int]:
            verts (Array): Hull vertices, shape (n, 2). The valid hull vertices
                are contained in the first num_valid rows
            num_valid (int): Number of valid hull vertices. Typically for a humanoid
                robot, this is 6 in general (or 4 if the feet are perfectly aligned
                with eachother)
    """

    def _cross_2d(o, a, b):
        """Z-component of (a-o) x (b-o). Positive = CCW turn."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    n = pts.shape[0]
    lex_key = pts[:, 1] * 1e6 + pts[:, 0]
    pivot_idx = jnp.argmin(lex_key)
    pivot = pts[pivot_idx]

    diff = pts - pivot
    angles = jnp.arctan2(diff[:, 1], diff[:, 0])
    angles = jnp.where(jnp.arange(n) == pivot_idx, jnp.inf, angles)

    order = jnp.argsort(angles)
    sorted_pts = pts[order]

    verts = jnp.zeros((n, 2))
    verts = verts.at[0].set(pivot)
    verts = verts.at[1].set(sorted_pts[0])
    num_valid = jnp.array(2)  # init

    for i in range(1, n - 1):  # unroll this loop
        p = sorted_pts[i]
        # NOTE: 2 is a good heuristic check-pop quantity for 8 points across 2 robot feet
        # TODO: Might need to pop upwards of num_valid - 2
        for _ in range(2):
            cross = _cross_2d(verts[num_valid - 2], verts[num_valid - 1], p)
            do_pop = (num_valid > 2) & (cross <= 1e-10)
            num_valid = jnp.where(do_pop, num_valid - 1, num_valid)
        verts = verts.at[num_valid].set(p)
        num_valid = num_valid + 1

    return verts, num_valid


def convex_hull_halfspaces(verts: Array, num_valid: int) -> tuple[Array, Array]:
    """After computing the convex hull result, convert it to halfspace form: Ax - b <= 0
    where a *negative* value for Ax - b implies being inside the hull.

    Equivalently, the distances to each face of the hull, for a point inside, can be
    written as b - Ax

    For any invalid hull faces, A and b include redundant rows where effectively we have
    0x <= 1 (always true)

    Args:
        verts (Array): Hull vertices, shape (n, 2). The valid hull vertices
            are contained in the first num_valid rows
        num_valid (int): Number of valid hull vertices. Typically for a humanoid
            robot, this is 6 in general (or 4 if the feet are perfectly aligned
            with eachother)

    Returns:
        tuple[Array, Array]:
            A (Array): Inequality matrix, shape (n, 2)
            b (Array): Inequality vector, shape (n,)
    """

    idx = jnp.arange(verts.shape[0])

    # Wrap the edges securely around `sp` (the last valid point connects back to 0)
    next_idx = (idx + 1) % num_valid
    next_verts = verts[next_idx]

    edges = next_verts - verts
    normals = jnp.stack([edges[:, 1], -edges[:, 0]], axis=1)

    # Calculate norms, strictly preventing division by zero on degenerate edges
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
    safe_norms = jnp.where(norms > 1e-7, norms, 1.0)

    A = normals / safe_norms
    b = jnp.sum(A * verts, axis=1)

    # Apply masking for constraints beyond the valid vertices
    mask = idx < num_valid

    # Invalid half-spaces become A=[0,0] and b=1 (0 <= 1, logically ignored)
    A = jnp.where(mask[:, None], A, jnp.zeros_like(A))
    b = jnp.where(mask, b, jnp.ones_like(b))

    return A, b


def rectangle_to_halfspaces_2d(pts: Array) -> tuple[Array, Array]:
    """Convert a known convex rectangle to halfspace form, Ax <= b

    Note: we do not add any padding to either A or b because we know we will
    always have 4 constraints. If padding is needed, this can be applied later

    Args:
        pts (Array): Points defining the vertices of the rectangle, shape (4, 2)
            Can be in any ordering

    Returns:
        tuple[Array, Array]:
            A (Array): Inequality matrix, shape (4, 2)
            b (Array): Inequality vector, shape (4,)
    """
    assert pts.shape == (4, 2)

    # Sort CCW by angle around centroid
    centroid = pts.mean(axis=0)
    angles = jnp.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = jnp.argsort(angles)
    verts = pts[order]  # (4, 2) CCW

    # Compute halfspaces from edges.
    next_verts = jnp.roll(verts, -1, axis=0)
    edges = next_verts - verts
    normals = jnp.stack([edges[:, 1], -edges[:, 0]], axis=1)  # outward, CCW
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
    A = normals / norms  # (4, 2) unit normals
    b = jnp.sum(A * verts, axis=1)  # (4,)

    return A, b


# @partial(jax.jit, static_argnames=("robot",))
def support_polygon(
    robot: Humanoid, q: Array, contact_mode: int
) -> Tuple[Array, Array]:
    """Compute the halfspace representation of the robot's support polygon,
    i.e. Ax <= b where negative values of Ax - b imply x is in the hull

    Note that due to the differing number of faces in the support polygon
    depending on the contact mode and feet placement, we pad A and b to
    a first-axis size of 8

    Args:
        robot (Humanoid): Robot to compute the support polygon for
        q (Array): Generalized coordinates, shape (n,)
        contact_mode (int): Current contact mode. 0: No contact,
            1: Left foot, 2: Right foot, 3: Both feet

    Returns:
        Tuple[Array, Array]:
            A (Array): Inequality matrix, shape (8, 2)
            b (Array): Inequality vector, shape (8,)
    """
    left_foot_points_xy = left_foot_support_points(robot, q)[:, :2]
    right_foot_points_xy = right_foot_support_points(robot, q)[:, :2]

    def _pad_single_foot(A, b):
        A_padded = jnp.zeros((8, 2))
        A_padded = A_padded.at[: A.shape[0]].set(A)
        b_padded = jnp.ones(8)
        b_padded = b_padded.at[: b.shape[0]].set(b)
        return A_padded, b_padded

    def _mode_0():  # No contact
        # TODO: Decide if this is the best return for this condition?
        # Currently: 0x <= -1 (always invalid)
        A = jnp.zeros((8, 2))
        b = -1 * jnp.ones(8)
        return A, b

    def _mode_1():  # Left foot
        pts = left_foot_points_xy
        A_unpadded, b_unpadded = rectangle_to_halfspaces_2d(pts)
        return _pad_single_foot(A_unpadded, b_unpadded)

    def _mode_2():  # Right foot
        pts = right_foot_points_xy
        A_unpadded, b_unpadded = rectangle_to_halfspaces_2d(pts)
        return _pad_single_foot(A_unpadded, b_unpadded)

    def _mode_3():  # Both feet
        pts = jnp.vstack([left_foot_points_xy, right_foot_points_xy])
        verts, num_valid = convex_hull_2d(pts)
        A, b = convex_hull_halfspaces(verts, num_valid)
        return A, b

    return jax.lax.switch(contact_mode, [_mode_0, _mode_1, _mode_2, _mode_3])
