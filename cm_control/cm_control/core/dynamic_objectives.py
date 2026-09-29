"""QP objective matrices for dynamics-based whole-body CBFs"""

import jax.numpy as jnp
from jax import Array
from frax import Humanoid

from cm_control.utils.osc_utils import (
    selection_matrix,
    contact_mask_from_mode,
    masked_dynamically_consistent_inverse,
)


"""
The following code assumes that we are going with the task hierarchy
structure as defined in the paper:
1. Contact forces
2. Motion tracking in task space
3. Posture tracking in the null space

and that the decision variable is just the actuated joint torques,
rather than the full generalized joint torques
"""


def P_qp(
    robot: Humanoid,
    q: Array,
    contact_mode: int,
    w_foot_force: Array,
    w_foot_motion: Array,
    w_hand: Array,
    w_com: Array,
    w_null: Array,
) -> Array:
    """Quadratic QP objective term for dynamics-based whole-body CBFs

    Note: this function is intended to be wrapped by another jitted
    function that explicitly marks robot and all of the weighting
    terms as static

    Args:
        robot (Humanoid): Robot to apply the filter to, with
            k actuated joints and n = k + 6 total DOFs
        q (Array): Generalized coordinates, shape (n,)
        contact_mode (int): Current contact mode. 0: No contact,
            1: Left foot, 2: Right foot, 3: Both feet
        w_foot_force (Array): Weights for a contact-force task
            for the feet, shape (6,)
        w_foot_motion (Array): Weights for a motion-tracking task
            for the feet, shape (6,)
        w_hand (Array): Weights for a motion-tracking task
            for the hands, shape (6,)
        w_com (Array): Weights for a (position-only) motion-tracking
            task for the COM, shape (3,)
        w_null (Array): Weights for a null space posture
            tracking task, shape (n,)

    Returns:
        Array: P_qp, shape (k, k)
    """
    # Get kinematics/dynamics terms
    S = selection_matrix(robot.num_actuated_joints)
    joint_transforms = robot.joint_to_world_transforms(q)
    J_lf = robot._left_foot_jacobian(joint_transforms)
    J_rf = robot._right_foot_jacobian(joint_transforms)
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)
    J_lh = robot._left_hand_jacobian(joint_transforms)
    J_rh = robot._right_hand_jacobian(joint_transforms)
    J_com = robot._center_of_mass_jacobian(joint_transforms)

    # Contact block: force tracking on any foot in contact
    # (up to 2 * 6 = 12 DOF)
    contact_mask = contact_mask_from_mode(contact_mode)
    w_c = contact_mask * jnp.concatenate([w_foot_force, w_foot_force])
    J_c = jnp.vstack([J_lf, J_rf])
    J_c_bar = masked_dynamically_consistent_inverse(J_c, contact_mask, M_inv)
    N_c = jnp.eye(robot.num_joints) - J_c_bar @ (contact_mask[:, None] * J_c)

    # Tracking block: motion tracking on any foot NOT in contact, plus
    # the hands and COM (up to 4 * 6 + 3 = 27 DOF)
    # Posture tracking in the remaining null space
    # TODO: CHECK IF THERE EVEN IS A NULLSPACE
    tracking_mask = jnp.concatenate([1.0 - contact_mask, jnp.ones(15)])
    w_t = tracking_mask * jnp.concatenate(
        [w_foot_motion, w_foot_motion, w_hand, w_hand, w_com]
    )
    J_t = jnp.vstack([J_lf, J_rf, J_lh, J_rh, J_com])
    J_tc = (tracking_mask[:, None] * J_t) @ N_c
    J_tc_bar = masked_dynamically_consistent_inverse(J_tc, tracking_mask, M_inv)
    N_tc = jnp.eye(robot.num_joints) - J_tc_bar @ J_tc

    contact_term_half = jnp.sqrt(w_c)[:, None] * (J_c_bar.T @ S.T)
    tracking_term_half = jnp.sqrt(w_t)[:, None] * (J_tc @ M_inv @ S.T)
    nullspace_term_half = jnp.sqrt(w_null)[:, None] * (M_inv @ N_tc.T @ S.T)

    stacked_terms_half = jnp.vstack(
        [contact_term_half, tracking_term_half, nullspace_term_half]
    )
    return stacked_terms_half.T @ stacked_terms_half


def q_qp_from_P(P: Array, u_des: Array) -> Array:
    """Linear QP objective term for dynamics-based whole-body CBFs

    This function assumes that we have already computed P, where
    most of the actual task hierarchy logic is performed

    Args:
        P (Array): QP quadratic term, shape (k, k)
        u_des (Array): Nominal actuated joint torques, shape (k,)

    Returns:
        Array: q_qp, shape (k,)
    """
    return -P @ u_des
