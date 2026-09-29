"""Test cases for dynamics objective terms"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import unittest
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import elastiqp.jax
from frax import load_g1

import cm_control.core.dynamic_objectives as dynamic_objectives
import cm_control.core.kinematic_objectives as kinematic_objectives

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("robot",))
def construct_dynamics_matrices(robot, z, u, contact_mode):
    w_foot_force = jnp.ones(6)
    w_foot_motion = jnp.ones(6)
    w_hand = jnp.ones(6)
    w_com = jnp.ones(3)
    w_null = jnp.ones(robot.num_joints)
    q = z[: robot.num_joints]
    P = dynamic_objectives.P_qp(
        robot, q, contact_mode, w_foot_force, w_foot_motion, w_hand, w_com, w_null
    )
    q = dynamic_objectives.q_qp_from_P(P, u)
    return P, q


@partial(jax.jit, static_argnames=("robot",))
def solve_dynamics_qp(robot, z, u, contact_mode):
    P, q = construct_dynamics_matrices(robot, z, u, contact_mode)
    # For now, (effectively) unconstrained QP (just testing the objective)
    G = jnp.ones((1, q.shape[0]))
    h = jnp.ones(1) * jnp.inf
    return elastiqp.jax.solve(P, q, G, h, penalty=1e3, eps_abs=1e-5)


@partial(jax.jit, static_argnames=("robot",))
def construct_kinematics_matrices(robot, z, u, contact_mode):
    w_foot_motion = jnp.ones(6)
    w_hand = jnp.ones(6)
    w_com = jnp.ones(3)
    w_null = jnp.ones(robot.num_joints)
    q = z[: robot.num_joints]
    P = kinematic_objectives.P_qp(
        robot, q, contact_mode, w_foot_motion, w_hand, w_com, w_null
    )
    q = kinematic_objectives.q_qp_from_P(P, u)
    return P, q


@partial(jax.jit, static_argnames=("robot",))
def solve_kinematics_qp(robot, z, u, contact_mode):
    P, q = construct_kinematics_matrices(robot, z, u, contact_mode)
    # For now, (effectively) unconstrained QP (just testing the objective)
    G = jnp.ones((1, q.shape[0]))
    h = jnp.ones(1) * jnp.inf
    return elastiqp.jax.solve(P, q, G, h, penalty=1e3, eps_abs=1e-5)


def is_spd(mat) -> bool:
    """Check that a matrix is symmetric positive definite

    Args:
        mat (Array): Matrix to check

    Returns:
        bool: True if the matrix is symmetric PD, False otherwise
    """
    mat = np.asarray(mat)
    # Must be square
    if mat.shape[0] != mat.shape[1]:
        return False
    # Must be symmetric or hermitian
    if not np.allclose(mat, mat.conj().T, atol=1e-14):
        return False
    # Check PD with cholesky
    try:
        np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        return False
    return True


@jax.tree_util.register_static
class TestDynamicsObjectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_g1()
        np.random.seed(0)

    # TODO: These tests right now just check to see if things run in the first place
    # We should test more actual properties of interest though

    def test_construction_dyn(self):
        num_tests = 10
        for _ in range(num_tests):
            z = np.random.rand(2 * self.robot.num_joints)
            u = np.random.rand(self.robot.num_actuated_joints)
            for contact_mode in [0, 1, 2, 3]:
                P, q = construct_dynamics_matrices(self.robot, z, u, contact_mode)

    def test_qp_solve_dyn(self):
        num_tests = 10
        for _ in range(num_tests):
            z = np.random.rand(2 * self.robot.num_joints)
            u = np.random.rand(self.robot.num_actuated_joints)
            for contact_mode in [0, 1, 2, 3]:
                x = solve_dynamics_qp(self.robot, z, u, contact_mode)

    def test_psd_dyn(self):
        num_tests = 10
        for _ in range(num_tests):
            z = np.random.rand(2 * self.robot.num_joints)
            u = np.random.rand(self.robot.num_actuated_joints)
            for contact_mode in [0, 1, 2, 3]:
                P, q = construct_dynamics_matrices(self.robot, z, u, contact_mode)
                assert is_spd(P)

    def test_construction_kin(self):
        num_tests = 10
        for _ in range(num_tests):
            z = np.random.rand(2 * self.robot.num_joints)
            u = np.random.rand(self.robot.num_joints)
            for contact_mode in [0, 1, 2, 3]:
                P, q = construct_kinematics_matrices(self.robot, z, u, contact_mode)

    def test_qp_solve_kin(self):
        num_tests = 10
        for _ in range(num_tests):
            z = np.random.rand(2 * self.robot.num_joints)
            u = np.random.rand(self.robot.num_joints)
            for contact_mode in [0, 1, 2, 3]:
                x = solve_kinematics_qp(self.robot, z, u, contact_mode)

    def test_psd_kin(self):
        num_tests = 10
        for _ in range(num_tests):
            z = np.random.rand(2 * self.robot.num_joints)
            u = np.random.rand(self.robot.num_joints)
            for contact_mode in [0, 1, 2, 3]:
                P, q = construct_kinematics_matrices(self.robot, z, u, contact_mode)
                assert is_spd(P)


if __name__ == "__main__":
    unittest.main()
