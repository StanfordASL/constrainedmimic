"""Quick vibe-coded validation plot to check that the support polygon logic is working"""

import numpy as np
import matplotlib.pyplot as plt

from cm_control.core.support_polygon import (
    convex_hull_2d,
    rectangle_to_halfspaces_2d,
    convex_hull_halfspaces,
)


def compute_halfspace_vertices(A, b, tol=1e-9):
    """
    Compute polygon vertices from halfspace representation:
        A x <= b

    Works by intersecting every pair of lines and checking feasibility.

    Args:
        A: (N,2)
        b: (N,)
        tol: feasibility tolerance

    Returns:
        verts: (M,2) ordered CCW
    """

    A = np.asarray(A)
    b = np.asarray(b)

    n = A.shape[0]

    candidates = []

    # Intersect every pair of constraints
    for i in range(n):
        for j in range(i + 1, n):
            Ai = A[i]
            Aj = A[j]

            M = np.stack([Ai, Aj])

            det = np.linalg.det(M)

            # Parallel lines
            if abs(det) < 1e-10:
                continue

            rhs = np.array([b[i], b[j]])

            x = np.linalg.solve(M, rhs)

            # Check feasibility
            if np.all(A @ x - b <= tol):
                candidates.append(x)

    if len(candidates) == 0:
        return np.zeros((0, 2))

    verts = np.unique(np.round(candidates, decimals=10), axis=0)

    # Sort CCW around centroid
    centroid = verts.mean(axis=0)

    angles = np.arctan2(
        verts[:, 1] - centroid[1],
        verts[:, 0] - centroid[0],
    )

    order = np.argsort(angles)

    return verts[order]


def plot_halfspaces(
    A,
    b,
    ax=None,
    xlim=(-1, 1),
    ylim=(-1, 1),
    show_constraints=True,
    show_normals=False,
    polygon_kwargs=None,
):
    """
    Plot a 2D polytope defined by:

        A x <= b

    Args:
        A: (N,2)
        b: (N,)
    """

    A = np.asarray(A)
    b = np.asarray(b)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if polygon_kwargs is None:
        polygon_kwargs = {}

    verts = compute_halfspace_vertices(A, b)

    # Plot polygon
    if len(verts) > 0:
        closed = np.vstack([verts, verts[0]])

        ax.fill(
            closed[:, 0],
            closed[:, 1],
            alpha=0.3,
            **polygon_kwargs,
        )

        ax.plot(
            closed[:, 0],
            closed[:, 1],
            linewidth=2,
        )

        ax.scatter(
            verts[:, 0],
            verts[:, 1],
            s=50,
        )

    # Plot each constraint line
    if show_constraints:
        xs = np.linspace(xlim[0], xlim[1], 500)

        for i in range(A.shape[0]):
            a1, a2 = A[i]
            bi = b[i]

            # Skip degenerate constraints
            if np.linalg.norm(A[i]) < 1e-10:
                continue

            if abs(a2) > 1e-10:
                ys = (bi - a1 * xs) / a2

                ax.plot(xs, ys, linestyle="--")

            else:
                x = bi / a1
                ax.axvline(x, linestyle="--")

            if show_normals:
                p = A[i] * bi

                ax.arrow(
                    p[0],
                    p[1],
                    0.1 * A[i, 0],
                    0.1 * A[i, 1],
                    head_width=0.02,
                    length_includes_head=True,
                )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal")
    ax.grid(True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Polytope: $Ax \leq b$")

    return ax


def rectangle_corners(center, yaw, length, width):
    """
    Compute rectangle corners from center pose.

    Args:
        center: (2,) array-like [x, y]
        yaw: rotation angle in radians
        length: foot length (x direction in local frame)
        width: foot width (y direction in local frame)

    Returns:
        corners: (4,2) array in CCW order
    """

    cx, cy = center

    hx = length / 2.0
    hy = width / 2.0

    # Local rectangle corners (CCW)
    local = np.array(
        [
            [hx, hy],
            [-hx, hy],
            [-hx, -hy],
            [hx, -hy],
        ]
    )

    c = np.cos(yaw)
    s = np.sin(yaw)

    R = np.array(
        [
            [c, -s],
            [s, c],
        ]
    )

    world = local @ R.T
    world += np.array([cx, cy])

    return world


def main():
    left_center = np.array([-0.05, 0.1])
    right_center = np.array([0.08, -0.08])
    foot_length = 0.17
    foot_width = 0.055
    left_yaw = np.deg2rad(10)
    right_yaw = np.deg2rad(-15)

    left_pts = rectangle_corners(
        left_center,
        left_yaw,
        foot_length,
        foot_width,
    )

    right_pts = rectangle_corners(
        right_center,
        right_yaw,
        foot_length,
        foot_width,
    )

    # A, b = rectangle_to_halfspaces_2d(left_pts)
    # A, b = rectangle_to_halfspaces_2d(right_pts)
    A, b = convex_hull_halfspaces(*convex_hull_2d(np.vstack([left_pts, right_pts])))

    plot_halfspaces(
        A,
        b,
        xlim=(-2, 2),
        ylim=(-2, 2),
        show_normals=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
