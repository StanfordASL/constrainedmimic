"""Test cases for control affine kinematics and dynamics"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import unittest
import time
from functools import partial

import numpy as np
import jax

from frax import load_g1
from cm_control.core.control_affine import (
    f_kin_simple,
    g_kin_simple,
    f_kin_constrained,
    g_kin_constrained,
    f_kin_constrained_underactuated,
    g_kin_constrained_underactuated,
    f_dyn_simple,
    g_dyn_simple,
    f_dyn_constrained,
    g_dyn_constrained,
    f_dyn_constrained_underactuated,
    g_dyn_constrained_underactuated,
)

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("robot",))
def simple_kinematics(robot, z, u):
    return f_kin_simple(robot, z) + g_kin_simple(robot, z) @ u


@partial(jax.jit, static_argnames=("robot",))
def constrained_kinematics(robot, z, u, contact_mode):
    return (
        f_kin_constrained(robot, z, contact_mode)
        + g_kin_constrained(robot, z, contact_mode) @ u
    )


@partial(jax.jit, static_argnames=("robot",))
def constrained_underactuated_kinematics(robot, z, u, contact_mode):
    return (
        f_kin_constrained_underactuated(robot, z, contact_mode)
        + g_kin_constrained_underactuated(robot, z, contact_mode) @ u
    )


@partial(jax.jit, static_argnames=("robot",))
def simple_dynamics(robot, z, u):
    return f_dyn_simple(robot, z) + g_dyn_simple(robot, z) @ u


@partial(jax.jit, static_argnames=("robot", "include_jdot"))
def constrained_dynamics(robot, z, u, contact_mode, include_jdot):
    return (
        f_dyn_constrained(robot, z, contact_mode, include_jdot)
        + g_dyn_constrained(robot, z, contact_mode, include_jdot) @ u
    )


@partial(jax.jit, static_argnames=("robot", "include_jdot"))
def constrained_underactuated_dynamics(robot, z, u, contact_mode, include_jdot):
    return (
        f_dyn_constrained_underactuated(robot, z, contact_mode, include_jdot)
        + g_dyn_constrained_underactuated(robot, z, contact_mode, include_jdot) @ u
    )


@jax.tree_util.register_static
class TestKinematicsAndDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_g1()
        cls.num_timing_tests = 100
        np.random.seed(0)

    def test_simple_kinematics(self):
        print("\n--- Simple kinematics ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_joints)
        z = q
        start_time_jit = time.perf_counter()
        z_dot = simple_kinematics(self.robot, z, u).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            z_dot = simple_kinematics(self.robot, z, u).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / self.num_timing_tests
        print("Average time (ms): ", avg_time * 1e3)

    def test_constrained_kinematics(self):
        print("\n--- Constrained kinematics ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_joints)
        z = q
        start_time_jit = time.perf_counter()
        jit_contact_mode = 3
        z_dot = constrained_kinematics(
            self.robot, z, u, jit_contact_mode
        ).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        contact_modes = (0, 1, 2, 3)
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            for contact_mode in contact_modes:
                z_dot = constrained_kinematics(
                    self.robot, z, u, contact_mode
                ).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / (self.num_timing_tests * len(contact_modes))
        print("Average time (ms): ", avg_time * 1e3)

    def test_constrained_underactuated_kinematics(self):
        print("\n--- Constrained underactuated kinematics ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_actuated_joints)
        z = q
        start_time_jit = time.perf_counter()
        jit_contact_mode = 3
        z_dot = constrained_underactuated_kinematics(
            self.robot, z, u, jit_contact_mode
        ).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        contact_modes = (0, 1, 2, 3)
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            for contact_mode in contact_modes:
                z_dot = constrained_underactuated_kinematics(
                    self.robot, z, u, contact_mode
                ).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / (self.num_timing_tests * len(contact_modes))
        print("Average time (ms): ", avg_time * 1e3)

    def test_simple_dynamics(self):
        print("\n--- Simple dynamics ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        qd = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_joints)
        z = np.concatenate([q, qd])
        start_time_jit = time.perf_counter()
        z_dot = simple_dynamics(self.robot, z, u).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            z_dot = simple_dynamics(self.robot, z, u).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / self.num_timing_tests
        print("Average time (ms): ", avg_time * 1e3)

    def test_constrained_dynamics(self):
        print("\n--- Constrained dynamics ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        qd = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_joints)
        z = np.concatenate([q, qd])
        start_time_jit = time.perf_counter()
        jit_contact_mode = 3
        include_jdot = False
        z_dot = constrained_dynamics(
            self.robot, z, u, jit_contact_mode, include_jdot
        ).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        contact_modes = (0, 1, 2, 3)
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            for contact_mode in contact_modes:
                z_dot = constrained_dynamics(
                    self.robot, z, u, contact_mode, include_jdot
                ).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / (self.num_timing_tests * len(contact_modes))
        print("Average time (ms): ", avg_time * 1e3)

    def test_constrained_underactuated_dynamics(self):
        print("\n--- Constrained underactuated dynamics ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        qd = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_actuated_joints)
        z = np.concatenate([q, qd])
        start_time_jit = time.perf_counter()
        jit_contact_mode = 3
        include_jdot = False
        z_dot = constrained_underactuated_dynamics(
            self.robot, z, u, jit_contact_mode, include_jdot
        ).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        contact_modes = (0, 1, 2, 3)
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            for contact_mode in contact_modes:
                z_dot = constrained_underactuated_dynamics(
                    self.robot, z, u, contact_mode, include_jdot
                ).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / (self.num_timing_tests * len(contact_modes))
        print("Average time (ms): ", avg_time * 1e3)

    def test_constrained_dynamics_jdot(self):
        print("\n--- Constrained dynamics [with Jdot terms] ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        qd = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_joints)
        z = np.concatenate([q, qd])
        start_time_jit = time.perf_counter()
        jit_contact_mode = 3
        include_jdot = True
        z_dot = constrained_dynamics(
            self.robot, z, u, jit_contact_mode, include_jdot
        ).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        contact_modes = (0, 1, 2, 3)
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            for contact_mode in contact_modes:
                z_dot = constrained_dynamics(
                    self.robot, z, u, contact_mode, include_jdot
                ).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / (self.num_timing_tests * len(contact_modes))
        print("Average time (ms): ", avg_time * 1e3)

    def test_constrained_underactuated_dynamics_jdot(self):
        print("\n--- Constrained underactuated dynamics [with Jdot terms] ---")

        # Initial call for JIT
        q = np.random.rand(self.robot.num_joints)
        qd = np.random.rand(self.robot.num_joints)
        u = np.random.rand(self.robot.num_actuated_joints)
        z = np.concatenate([q, qd])
        start_time_jit = time.perf_counter()
        jit_contact_mode = 3
        include_jdot = True
        z_dot = constrained_underactuated_dynamics(
            self.robot, z, u, jit_contact_mode, include_jdot
        ).block_until_ready()
        jit_time = time.perf_counter() - start_time_jit
        print("JIT time (ms): ", jit_time * 1e3)

        # Time a batch of evals
        contact_modes = (0, 1, 2, 3)
        start_time_batch = time.perf_counter()
        for _ in range(self.num_timing_tests):
            for contact_mode in contact_modes:
                z_dot = constrained_underactuated_dynamics(
                    self.robot, z, u, contact_mode, include_jdot
                ).block_until_ready()
        batch_time = time.perf_counter() - start_time_batch
        avg_time = batch_time / (self.num_timing_tests * len(contact_modes))
        print("Average time (ms): ", avg_time * 1e3)


if __name__ == "__main__":
    unittest.main()
