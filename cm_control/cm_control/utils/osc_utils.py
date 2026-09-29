"""Utilities for operational space control"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

# Number of constraint rows per foot contact (6-DOF flat contact)
DOFS_PER_FOOT_CONTACT = 6


def nullspace_projection(J: Array, M_inv: Array) -> Array:
    """Compute the (dynamically-consistent) null space projection matrix
    for a given Jacobian

    NOTE: If the robot is near a kinematic singularity, this can lead to
    numerical instability. Consider using the `stable_` version instead

    Args:
        J (Array): Jacobian, shape (m, n)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: N, shape (n, n)
    """
    J_bar = dynamically_consistent_inverse(J, M_inv)
    return nullspace_projection_from_inverse(J, J_bar)


def nullspace_projection_from_inverse(J: Array, J_bar: Array) -> Array:
    """Compute the (dynamically-consistent) null space projection matrix
    for a given Jacobian, from an already-computed Jacobian inverse

    Args:
        J (Array): Jacobian, shape (m, n)
        J_bar (Array): Jacobian inverse (typically, dynamically-consistent),
            shape (n, m)

    Returns:
        Array: Null space projection, shape (n, n)
    """
    nq = J.shape[1]
    return jnp.eye(nq) - J_bar @ J


def selection_matrix(k: int) -> Array:
    """Actuation selection matrix for a floating-base robot

    Args:
        k (int): Number of actuated DOFs

    Returns:
        Array: S, shape (k, k + 6)
    """
    return jnp.hstack([jnp.zeros((k, 6)), jnp.eye(k)])


def dynamically_consistent_inverse(J: Array, M_inv: Array) -> Array:
    """Compute the dynamically-consistent Jacobian inverse

    Args:
        J (Array): Jacobian, shape (m, n)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: J_bar, shape (n, m)
    """
    Lambda = task_space_inertia(J, M_inv)
    return M_inv @ J.T @ Lambda


def task_space_jdot_term(qd: Array, J: Array, J_dot: Array, M_inv: Array) -> Array:
    """Compute the component of the task-space centrifugal/
    coriolis forces associated with the Jacobian's time derivative

    Args:
        qd (Array): Joint velocities, shape (n,)
        J (Array): Jacobian, shape (m, n)
        J_dot (Array): Jacobian time derivative, shape (m, n)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: mu_c_jdot, shape (m,)
    """
    Lambda = task_space_inertia(J, M_inv)
    return -Lambda @ J_dot @ qd


def task_space_inertia(J: Array, M_inv: Array) -> Array:
    """Compute the task-space inertia matrix

    Args:
        J (Array): Jacobian, shape (m, n)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: Lambda, shape (m, m)
    """
    return jnp.linalg.inv(J @ M_inv @ J.T)


def underactuated_contact_nullspace_projection(
    J_c: Array, M_inv: Array, S: Array
) -> Array:
    """Compute the projection matrix mapping from actuated
    joint velocites to generalized joint velocities

    NOTE: If the robot is near a kinematic singularity, this can lead to
    numerical instability. Consider using the `stable_` version instead

    Args:
        J_c (Array): Contact Jacobian, shape (m, n)
        M_inv (Array): Inverse mass matrix, shape (n, n)
        S (Array): Selection matrix, shape (k, n)

    Returns:
        Array: SN_c_bar, shape (n, k)
    """
    N_c = nullspace_projection(J_c, M_inv)
    SN_c = S @ N_c
    return M_inv @ SN_c.T @ jnp.linalg.pinv(SN_c @ M_inv @ SN_c.T)


###########################
# MASKED CONTACT HANDLING
###########################

# Essentially, the gist of this masked handling is that
# instead of dealing with contact jacobians that are sometimes
# (6, nq), sometimes (12, nq), and technically also (0, nq),
# we just always handle the (12, nq) case and mask out any entries
# that are not needed for the actual contact jacobian, given
# the contact mode. This eliminates the need for a jax.lax.switch
# statement and keeps everything branch-free


def contact_mask_from_mode(contact_mode: int) -> Array:
    """Convert a contact mode to a binary mask over the rows of the stacked
    [left; right] foot contact Jacobian

    Args:
        contact_mode (int): Current contact mode. 0: No contact,
            1: Left foot, 2: Right foot, 3: Both feet.

    Returns:
        Array: Contact mask, shape (12,)
    """

    def _contact_state(contact_mode: int) -> Array:
        # 0/1 for left/right foot if in contact. shape (2,)
        # NOTE: the bits must be extracted from the integer mode BEFORE any
        # cast to bool/float (a bool cast maps all nonzero modes to 1)
        mode = jnp.asarray(contact_mode)
        return jnp.stack([mode & 1, (mode >> 1) & 1])

    return jnp.repeat(_contact_state(contact_mode), 6).astype(float)


def masked_task_space_inertia(J: Array, mask: Array, M_inv: Array) -> Array:
    """Task-space inertia matrix for the active rows of a Jacobian

    Acts as identity on the masked-out rows/columns

    Args:
        J (Array): Jacobian, shape (m, n)
        mask (Array): 0/1 mask over the rows of J, shape (m,)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: Lambda, shape (m, m)
    """
    J_m = mask[:, None] * J
    # Pad with identity on the masked rows to maintain invertibility
    return jnp.linalg.inv(J_m @ M_inv @ J_m.T + jnp.diag(1.0 - mask))


def masked_dynamically_consistent_inverse(J: Array, mask: Array, M_inv: Array) -> Array:
    """Dynamically-consistent inverse of the mask-active rows of a Jacobian

    Columns corresponding to masked-out rows are exactly zero

    Args:
        J (Array): Jacobian, shape (m, n)
        mask (Array): 0/1 mask over the rows of J, shape (m,)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: J_bar, shape (n, m)
    """
    J_m = mask[:, None] * J
    Lambda = masked_task_space_inertia(J, mask, M_inv)
    return M_inv @ J_m.T @ Lambda


def masked_nullspace_projection(J: Array, mask: Array, M_inv: Array) -> Array:
    """Dynamically-consistent null space projection for the mask-active
    rows of a Jacobian

    With an all-zero mask, this is the identity (no constraint)

    Args:
        J (Array): Jacobian, shape (m, n)
        mask (Array): 0/1 mask over the rows of J, shape (m,)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: N, shape (n, n)
    """
    J_m = mask[:, None] * J
    J_bar = masked_dynamically_consistent_inverse(J, mask, M_inv)
    return jnp.eye(J.shape[1]) - J_bar @ J_m


def masked_task_space_jdot_term(
    qd: Array, J: Array, J_dot: Array, mask: Array, M_inv: Array
) -> Array:
    """Task-space centrifugal/coriolis jdot term for the mask-active
    rows of a Jacobian. Masked-out rows are exactly zero

    Args:
        qd (Array): Joint velocities, shape (n,)
        J (Array): Jacobian, shape (m, n)
        J_dot (Array): Jacobian time derivative, shape (m, n)
        mask (Array): 0/1 mask over the rows of J, shape (m,)
        M_inv (Array): Inverse mass matrix, shape (n, n)

    Returns:
        Array: mu_c_jdot, shape (m,)
    """
    Lambda = masked_task_space_inertia(J, mask, M_inv)
    return -Lambda @ ((mask[:, None] * J_dot) @ qd)


def masked_underactuated_contact_nullspace_projection(
    J_c: Array, mask: Array, M_inv: Array, S: Array
) -> Array:
    """Projection matrix mapping from actuated joint velocities to
    generalized joint velocities, for the mask-active rows of a
    contact Jacobian

    Args:
        J_c (Array): Contact Jacobian, shape (m, n)
        mask (Array): 0/1 mask over the rows of J_c, shape (m,)
        M_inv (Array): Inverse mass matrix, shape (n, n)
        S (Array): Selection matrix, shape (k, n)

    Returns:
        Array: SN_c_bar, shape (n, k)
    """
    N_c = masked_nullspace_projection(J_c, mask, M_inv)
    SN_c = S @ N_c
    return M_inv @ SN_c.T @ jnp.linalg.pinv(SN_c @ M_inv @ SN_c.T)


###########################
# ADVANCED STUFF
# (for numerical stability)
###########################


def stable_dynamically_consistent_inverse(
    J: Array, M: Array, is_gpu: bool, is_x64: bool
) -> Array:
    """Numerically stable dynamically-consistent Jacobian inverse"""
    # GPU is not very good at SVD/pinv so we use eigh instead
    # But, eigh is slightly less numerically stable than pinv since
    # computing the gram matrix squares the condition number
    if is_gpu:
        rcond = 1e-6 if is_x64 else 1e-3
        return _stable_dynamically_consistent_inverse_eigh(J, M, rcond)
    # If we are on CPU, we can handle much tighter numerical tolerances
    # and use pinv just fine
    else:
        rcond = 1e-12 if is_x64 else 1e-5
        return _stable_dynamically_consistent_inverse_pinv(J, M, rcond)


def _stable_dynamically_consistent_inverse_pinv(
    J: Array, M: Array, rcond: float
) -> Array:
    # Factorize mass matrix
    L = jsp.linalg.cholesky(M, lower=True)
    # Project Jacobian into inertially-weighted space: J_tilde = J L^-T
    J_tilde = jsp.linalg.solve_triangular(L, J.T, lower=True).T
    # Compute pinv in the scaled coordinate space
    J_tilde_pinv = jnp.linalg.pinv(J_tilde, rtol=rcond)
    # Transform back into the original space: J_bar_stable = L^-T @ pinv(J_tilde)
    J_bar_stable = jsp.linalg.solve_triangular(L.T, J_tilde_pinv, lower=False)
    # If full rank, this is equivalent to M_inv @ J.T @ inv(J @ M_inv @ J.T)
    return J_bar_stable


def _stable_dynamically_consistent_inverse_eigh(
    J: Array, M: Array, rcond: float
) -> Array:
    # Factorize mass matrix
    L = jsp.linalg.cholesky(M, lower=True)
    # Project Jacobian into inertially-weighted space: J_tilde = J L^-T
    J_tilde = jsp.linalg.solve_triangular(L, J.T, lower=True).T
    # Compute Gram matrix (symmetric PSD)
    G = J_tilde @ J_tilde.T
    # Eigendecomposition of Gram matrix
    w, V = jnp.linalg.eigh(G)
    # Zero out eigenvals below a rcond**2 threshold (eigenvals are singular vals**2)
    keep = w > (rcond * rcond) * w[-1]
    inv_w = jnp.where(keep, 1.0 / w, 0.0)
    # Reconstruct the pseudoinverse from the updated inverse eigenvals
    J_tilde_pinv = J_tilde.T @ (V * inv_w) @ V.T
    return jsp.linalg.solve_triangular(L.T, J_tilde_pinv, lower=False)


def stable_nullspace_projection(
    J: Array, M: Array, is_gpu: bool, is_x64: bool
) -> Array:
    """Numerically stable dynamically-consistent null space projection matrix"""
    J_bar = stable_dynamically_consistent_inverse(J, M, is_gpu, is_x64)
    return nullspace_projection_from_inverse(J, J_bar)


def stable_underactuated_contact_nullspace_projection(
    J_c: Array, M: Array, S: Array, is_gpu: bool, is_x64: bool
) -> Array:
    N_c = stable_nullspace_projection(J_c, M, is_gpu, is_x64)
    return stable_dynamically_consistent_inverse(S @ N_c, M, is_gpu, is_x64)


def _main():
    # my assorted testing
    import numpy as np
    import jax

    jax.config.update("jax_enable_x64", True)

    np.random.seed(0)
    Jl = np.random.rand(6, 10)
    Jr = np.random.rand(6, 10)
    contact_mode = 1
    M = np.random.rand(10, 10)
    M = M @ M.T
    M_inv = np.linalg.inv(M)
    J = np.vstack([Jl, Jr])
    L1 = task_space_inertia(Jl, M_inv)

    L2 = masked_task_space_inertia(J, contact_mask_from_mode(contact_mode), M_inv)
    print(np.max(np.abs(L2[:6, :6] - L1)))


if __name__ == "__main__":
    _main()
