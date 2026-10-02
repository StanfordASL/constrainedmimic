"""Tests for the numerically-stable OSC utilities in ``osc_utils``.

NOTE: Claude-generated tests. Seem reasonable, though
"""

import os

# Run tests on CPU
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest

import numpy as np
import jax
import jax.numpy as jnp

from frax import load_g1
from cm_control.utils.osc_utils import (
    nullspace_projection,
    dynamically_consistent_inverse,
    underactuated_contact_nullspace_projection,
    stable_nullspace_projection,
    stable_dynamically_consistent_inverse,
    stable_underactuated_contact_nullspace_projection,
    selection_matrix,
)

# (is_gpu, is_x64) combinations. is_x64 is True to match the global config; the
# two entries exercise the pinv (CPU) and eigh (GPU) code paths.
COMPUTE_PATHS = [(False, True), (True, True)]
CONTACT_MODES = (1, 2, 3)  # 1=left foot, 2=right foot, 3=both


class TestStableOscUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_g1()
        cls.n = cls.robot.num_joints
        cls.k = cls.robot.num_actuated_joints
        np.random.seed(0)

    def _contact_jacobian(self, contact_mode):
        """Contact Jacobian + mass matrix at a perturbed standing config."""
        q = np.zeros(self.n)
        q[2] = 0.75  # root height
        q[6:] = 0.05 * np.random.randn(self.k)
        jt = self.robot.joint_to_world_transforms(q)
        J_lf = self.robot._left_foot_jacobian(jt)
        J_rf = self.robot._right_foot_jacobian(jt)
        M = self.robot._mass_matrix(jt)
        J = {1: J_lf, 2: J_rf, 3: jnp.vstack([J_lf, J_rf])}[contact_mode]
        return J, M

    # -- Equivalence to the naive DC results on real (well-conditioned) J ------
    def test_projection_reduces_to_naive_dc(self):
        """stable_nullspace_projection == the naive DC projector when nothing is
        truncated, on both compute paths and every contact mode."""
        for contact_mode in CONTACT_MODES:
            J, M = self._contact_jacobian(contact_mode)
            M_inv = self.robot.mass_matrix_inverse(M)
            N_naive = np.asarray(nullspace_projection(J, M_inv))
            for is_gpu, is_x64 in COMPUTE_PATHS:
                with self.subTest(contact_mode=contact_mode, is_gpu=is_gpu):
                    N_stable = np.asarray(
                        stable_nullspace_projection(J, M, is_gpu, is_x64)
                    )
                    np.testing.assert_allclose(N_stable, N_naive, rtol=0, atol=1e-8)

    def test_dc_inverse_reduces_to_naive(self):
        """stable_dynamically_consistent_inverse == the naive DC inverse when
        nothing is truncated."""
        for contact_mode in CONTACT_MODES:
            J, M = self._contact_jacobian(contact_mode)
            M_inv = self.robot.mass_matrix_inverse(M)
            Jbar_naive = np.asarray(dynamically_consistent_inverse(J, M_inv))
            for is_gpu, is_x64 in COMPUTE_PATHS:
                with self.subTest(contact_mode=contact_mode, is_gpu=is_gpu):
                    Jbar_stable = np.asarray(
                        stable_dynamically_consistent_inverse(J, M, is_gpu, is_x64)
                    )
                    self.assertEqual(Jbar_stable.shape, (self.n, J.shape[0]))
                    np.testing.assert_allclose(
                        Jbar_stable, Jbar_naive, rtol=0, atol=1e-8
                    )

    def test_underactuated_reduces_to_naive(self):
        """stable_underactuated_contact_nullspace_projection == the original
        (M_inv-based) underactuated projection, with the expected (n, k) shape."""
        S = selection_matrix(self.k)
        for contact_mode in CONTACT_MODES:
            J, M = self._contact_jacobian(contact_mode)
            M_inv = self.robot.mass_matrix_inverse(M)
            naive = np.asarray(underactuated_contact_nullspace_projection(J, M_inv, S))
            for is_gpu, is_x64 in COMPUTE_PATHS:
                with self.subTest(contact_mode=contact_mode, is_gpu=is_gpu):
                    stable = np.asarray(
                        stable_underactuated_contact_nullspace_projection(
                            J, M, S, is_gpu, is_x64
                        )
                    )
                    self.assertEqual(stable.shape, (self.n, self.k))
                    np.testing.assert_allclose(stable, naive, rtol=0, atol=1e-7)

    # -- Projector properties --------------------------------------------------
    def test_projector_properties(self):
        """N is an idempotent projector (N^2 = N) onto the constraint null space
        (J N = 0), so a projected velocity produces no contact-foot motion."""
        for contact_mode in CONTACT_MODES:
            with self.subTest(contact_mode=contact_mode):
                J, M = self._contact_jacobian(contact_mode)
                N = np.asarray(stable_nullspace_projection(J, M, False, True))
                np.testing.assert_allclose(N @ N, N, rtol=0, atol=1e-9)
                v = np.random.randn(self.n)
                foot_twist = np.asarray(J) @ (N @ v)
                np.testing.assert_allclose(foot_twist, 0.0, rtol=0, atol=1e-8)

    # -- Stability: the reason the stable variant exists -----------------------
    def test_dc_inverse_bounded_at_singularity(self):
        """On a near-rank-deficient Jacobian the naive DC inverse explodes while
        the stable one truncates the tiny direction and stays bounded."""
        n, m = self.n, 6
        U, _ = np.linalg.qr(np.random.randn(m, m))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e-7])  # one near-zero direction
        J = jnp.asarray(U @ np.diag(s) @ V[:m])
        M = jnp.eye(n)
        M_inv = jnp.eye(n)

        naive = float(
            np.linalg.norm(np.asarray(dynamically_consistent_inverse(J, M_inv)), 2)
        )
        stable = float(
            np.linalg.norm(
                np.asarray(stable_dynamically_consistent_inverse(J, M, True, True)), 2
            )
        )
        self.assertGreater(naive, 1e5)
        self.assertLess(stable, 10.0)

    # -- Runs under jit (lives in the JIT-compiled integrate step) -------------
    def test_jittable(self):
        S = selection_matrix(self.k)
        J, M = self._contact_jacobian(contact_mode=3)

        N = jax.jit(stable_nullspace_projection, static_argnums=(2, 3))(
            J, M, False, True
        )
        self.assertEqual(N.shape, (self.n, self.n))
        self.assertFalse(bool(jnp.any(jnp.isnan(N))))

        SN = jax.jit(
            stable_underactuated_contact_nullspace_projection, static_argnums=(3, 4)
        )(J, M, jnp.asarray(S), False, True)
        self.assertEqual(SN.shape, (self.n, self.k))
        self.assertFalse(bool(jnp.any(jnp.isnan(SN))))


if __name__ == "__main__":
    unittest.main()
