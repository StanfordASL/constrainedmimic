"""Control-Affine Kinematics and Dynamics

For a given state z, the control affine form of the state ODE is
```
z_dot = f(z) + g(z) u
```
given some (possibly nonlinear) functions f(z) and g(z)

For kinematics / velocity-control, z = the joint state, [q]

For dynamics / torque-control, z = the joint state and velocity, [q, qd]

With a humanoid robot, we can model the kinematics and dynamics in a few
ways, of varying accuracy

1) Simplest form: Assume fully-actuated control over all DOFs (including
the floating base DOFs), and no contact constraints

2) Constrained form: Still assume full actuation over all DOFs, but now
project the kinematics/dynamics into the null space of the constraints

3) Constrained + underactuated form: Now, assume control over only the
actuated DOFs, and project into the constraint null space
"""

import jax.numpy as jnp
from jax import Array

from frax import Humanoid

from cm_control.utils.osc_utils import (
    selection_matrix,
    contact_mask_from_mode,
    masked_nullspace_projection,
    masked_task_space_jdot_term,
    masked_underactuated_contact_nullspace_projection,
)

"""
Simple Kinematics
z = q_dot
u = q_dot
z_dot = q_dot
"""


def f_kin_simple(robot: Humanoid, z: Array) -> Array:
    return jnp.zeros(robot.num_joints)


def g_kin_simple(robot: Humanoid, z: Array) -> Array:
    return jnp.eye(robot.num_joints)


"""
Constrained Kinematics
z = q_dot
u = q_dot
z_dot = Nc q_dot
where Nc is the contact null space projection matrix
"""


def f_kin_constrained(robot: Humanoid, z: Array, contact_mode: int) -> Array:
    return jnp.zeros(robot.num_joints)


def g_kin_constrained(robot: Humanoid, z: Array, contact_mode: int) -> Array:
    # Unpack state variable
    q = z

    # Get kinematics/dynamics terms
    joint_transforms = robot.joint_to_world_transforms(q)
    J_c_L = robot._left_foot_jacobian(joint_transforms)
    J_c_R = robot._right_foot_jacobian(joint_transforms)
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)

    mask = contact_mask_from_mode(contact_mode)
    return masked_nullspace_projection(jnp.vstack([J_c_L, J_c_R]), mask, M_inv)


"""
Constrained, Underactuated Kinematics
z = q_dot
u = q_dot_actuated
z_dot = (SNC)_bar q_dot_actuated
where Nc is the contact null space projection matrix, 
S is the selection matrix for the underactuated DOFs,
and bar represents the dynamically-consistent inverse
"""


def f_kin_constrained_underactuated(
    robot: Humanoid, z: Array, contact_mode: int
) -> Array:
    return jnp.zeros(robot.num_joints)


def g_kin_constrained_underactuated(
    robot: Humanoid, z: Array, contact_mode: int
) -> Array:
    # Unpack state variable
    q = z

    # Get kinematics/dynamics terms
    joint_transforms = robot.joint_to_world_transforms(q)
    J_c_L = robot._left_foot_jacobian(joint_transforms)
    J_c_R = robot._right_foot_jacobian(joint_transforms)
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)

    S = selection_matrix(robot.num_actuated_joints)

    mask = contact_mask_from_mode(contact_mode)
    return masked_underactuated_contact_nullspace_projection(
        jnp.vstack([J_c_L, J_c_R]), mask, M_inv, S
    )


"""
Simple dynamics
z = [q, qdot]
u = [tau]
z_dot = [qdot, M_inv (tau - c - g)]

This is the same as what was used in OSCBF as we did not handle
underactuation or contact constraints here (not necessary for a
fixed-base robot arm moving in free space)
"""


def f_dyn_simple(robot: Humanoid, z: Array) -> Array:
    # Unpack state variable
    q = z[: robot.num_joints]
    qd = z[robot.num_joints :]

    # Get kinematics/dynamics terms
    joint_transforms = robot.joint_to_world_transforms(q)
    bias = robot._nonlinear_bias(qd, joint_transforms)  # c + g
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)

    return jnp.concatenate([qd, -M_inv @ bias])


def g_dyn_simple(robot: Humanoid, z: Array) -> Array:
    # Unpack state variable
    q = z[: robot.num_joints]

    # Get kinematics/dynamics terms
    joint_transforms = robot.joint_to_world_transforms(q)
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)

    return jnp.vstack([jnp.zeros((robot.num_joints, robot.num_joints)), M_inv])


"""
Constrained dynamics
z = [q, qdot]
u = [tau]
z_dot = [qdot, Minv @ (Nc.T @ tau - N_c.T @ bias - J_c.T @ mu_c_jdot)]

Note that in the expression for z_dot, the contact terms only show up in
the qdd component, since the resultant integrated velocity should still
be consistent with the contact constraints without additional projection
"""


# include_jdot MUST BE STATIC
def f_dyn_constrained(
    robot: Humanoid, z: Array, contact_mode: int, include_jdot: bool
) -> Array:
    # Unpack state variable
    q = z[: robot.num_joints]
    qd = z[robot.num_joints :]

    # Get kinematics/dynamics terms
    joint_transforms = robot.joint_to_world_transforms(q)
    if include_jdot:
        J_c_L, J_c_L_dot = robot._left_foot_jacobian_and_derivative(
            qd, joint_transforms
        )
        J_c_R, J_c_R_dot = robot._right_foot_jacobian_and_derivative(
            qd, joint_transforms
        )
    else:
        J_c_L = robot._left_foot_jacobian(joint_transforms)
        J_c_R = robot._right_foot_jacobian(joint_transforms)
    bias = robot._nonlinear_bias(qd, joint_transforms)  # c + g
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)

    # f = [qdot, M_inv @ (-N_c.T @ bias - J_c.T @ mu_c_jdot)]
    # With no contact, N_c = identity and mu_c_jdot = 0, reducing to
    # the simple dynamics [qdot, -M_inv @ bias]
    mask = contact_mask_from_mode(contact_mode)
    J_c = jnp.vstack([J_c_L, J_c_R])
    N_c = masked_nullspace_projection(J_c, mask, M_inv)
    if include_jdot:
        J_c_dot = jnp.vstack([J_c_L_dot, J_c_R_dot])
        mu_c_jdot = masked_task_space_jdot_term(qd, J_c, J_c_dot, mask, M_inv)
    else:
        mu_c_jdot = jnp.zeros(J_c.shape[0])
    J_c_masked = mask[:, None] * J_c
    return jnp.concatenate([qd, M_inv @ (-N_c.T @ bias - J_c_masked.T @ mu_c_jdot)])


def g_dyn_constrained(
    robot: Humanoid, z: Array, contact_mode: int, include_jdot: bool
) -> Array:
    # Unpack state variable
    q = z[: robot.num_joints]

    # Get kinematics/dynamics terms
    joint_transforms = robot.joint_to_world_transforms(q)
    J_c_L = robot._left_foot_jacobian(joint_transforms)
    J_c_R = robot._right_foot_jacobian(joint_transforms)
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)

    # g = [0, M_inv @ N_c.T]
    # With no contact, N_c = identity, reducing to the simple dynamics
    mask = contact_mask_from_mode(contact_mode)
    N_c = masked_nullspace_projection(jnp.vstack([J_c_L, J_c_R]), mask, M_inv)

    zeros = jnp.zeros((robot.num_joints, robot.num_joints))
    return jnp.vstack([zeros, M_inv @ N_c.T])


"""
Constrained, underactuated dynamics
z = [q, qdot]
u = [tau_actuated]
z_dot = [qdot, Minv @ (Nc.T @ S.T @ tau - N_c.T @ bias - J_c.T @ mu_c_jdot)]

This is VERY similar to the constrained, fully-actuated case except now in
the g function, we have a component depending on S
"""


# include_jdot MUST BE STATIC
def f_dyn_constrained_underactuated(
    robot: Humanoid, z: Array, contact_mode: int, include_jdot: bool
) -> Array:
    # The autonomous dynamics are the same as the fully actuated case
    return f_dyn_constrained(robot, z, contact_mode, include_jdot)


def g_dyn_constrained_underactuated(
    robot: Humanoid, z: Array, contact_mode: int, include_jdot: bool
) -> Array:
    # Unpack state variable
    q = z[: robot.num_joints]

    # Get kinematics/dynamics terms
    joint_transforms = robot.joint_to_world_transforms(q)
    J_c_L = robot._left_foot_jacobian(joint_transforms)
    J_c_R = robot._right_foot_jacobian(joint_transforms)
    M = robot._mass_matrix(joint_transforms)
    M_inv = robot.mass_matrix_inverse(M)

    S = selection_matrix(robot.num_actuated_joints)

    # g = [0, M_inv @ N_c.T @ S.T]
    # With no contact, N_c = identity, giving [0, M_inv @ S.T]
    mask = contact_mask_from_mode(contact_mode)
    N_c = masked_nullspace_projection(jnp.vstack([J_c_L, J_c_R]), mask, M_inv)

    zeros = jnp.zeros((robot.num_joints, robot.num_actuated_joints))
    return jnp.vstack([zeros, M_inv @ N_c.T @ S.T])
