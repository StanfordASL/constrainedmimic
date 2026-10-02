"""Tests for the masked (switch-free) contact handling in ``osc_utils``.

The masked utilities operate on the full stacked [left; right] foot Jacobian
with a 0/1 row mask, and must EXACTLY reproduce the unmasked computation on
the active rows alone (identity padding on the masked-out block).

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

from cm_control.utils.osc_utils import (
    task_space_inertia,
    dynamically_consistent_inverse,
    nullspace_projection,
    task_space_jdot_term,
    underactuated_contact_nullspace_projection,
    selection_matrix,
    contact_mask_from_mode,
    masked_task_space_inertia,
    masked_dynamically_consistent_inverse,
    masked_nullspace_projection,
    masked_task_space_jdot_term,
    masked_underactuated_contact_nullspace_projection,
)

# Active row slice of the stacked [left; right] Jacobian, per contact mode
ACTIVE_ROWS = {1: slice(0, 6), 2: slice(6, 12), 3: slice(0, 12)}
CONTACT_MODES = (1, 2, 3)  # 1=left foot, 2=right foot, 3=both


class TestMaskedOscUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 16
        cls.k = cls.n - 6  # actuated DOFs (selection_matrix assumes n = k + 6)
        np.random.seed(0)
        cls.J = np.random.rand(12, cls.n)  # stacked [left; right] Jacobian
        cls.J_dot = np.random.rand(12, cls.n)
        cls.qd = np.random.rand(cls.n)
        A = np.random.randn(cls.n, cls.n)
        M = A @ A.T + cls.n * np.eye(cls.n)  # well-conditioned SPD mass matrix
        cls.M_inv = np.linalg.inv(M)
        cls.S = selection_matrix(cls.k)

    # -- The mode -> mask conversion (all 4 modes) ------------------------------
    def test_contact_mask_from_mode(self):
        """Left foot mask is the first 6 rows, right foot the last 6."""
        expected = {
            0: [0, 0],
            1: [1, 0],
            2: [0, 1],
            3: [1, 1],
        }
        for mode, (left, right) in expected.items():
            with self.subTest(contact_mode=mode):
                mask = np.asarray(contact_mask_from_mode(mode))
                self.assertEqual(mask.shape, (12,))
                np.testing.assert_array_equal(mask[:6], left)
                np.testing.assert_array_equal(mask[6:], right)
                # Also under jit with a traced mode (as used in the filters)
                mask_jit = np.asarray(jax.jit(contact_mask_from_mode)(mode))
                np.testing.assert_array_equal(mask_jit, mask)

    # -- Masked == unmasked on the active block, for every contact mode --------
    def test_masked_task_space_inertia(self):
        """Active block matches the unmasked Lambda; padded block is identity"""
        for mode in CONTACT_MODES:
            with self.subTest(contact_mode=mode):
                rows = ACTIVE_ROWS[mode]
                mask = contact_mask_from_mode(mode)
                L = np.asarray(masked_task_space_inertia(self.J, mask, self.M_inv))
                L_ref = np.asarray(task_space_inertia(self.J[rows], self.M_inv))
                np.testing.assert_allclose(L[rows, rows], L_ref, rtol=0, atol=1e-12)
                # Everything outside the active block is identity
                L_expected = np.eye(12)
                L_expected[rows, rows] = L_ref
                np.testing.assert_allclose(L, L_expected, rtol=0, atol=1e-12)

    def test_masked_task_space_inertia_no_contact(self):
        """With no contact, Lambda is the (padded) identity"""
        mask = contact_mask_from_mode(0)
        L = np.asarray(masked_task_space_inertia(self.J, mask, self.M_inv))
        np.testing.assert_allclose(L, np.eye(12), rtol=0, atol=1e-15)

    def test_masked_dynamically_consistent_inverse(self):
        """Active columns match the unmasked J_bar; masked columns are zero"""
        for mode in CONTACT_MODES:
            with self.subTest(contact_mode=mode):
                rows = ACTIVE_ROWS[mode]
                mask = np.asarray(contact_mask_from_mode(mode))
                J_bar = np.asarray(
                    masked_dynamically_consistent_inverse(self.J, mask, self.M_inv)
                )
                J_bar_ref = np.asarray(
                    dynamically_consistent_inverse(self.J[rows], self.M_inv)
                )
                np.testing.assert_allclose(
                    J_bar[:, rows], J_bar_ref, rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(J_bar[:, mask == 0], 0.0, rtol=0, atol=1e-15)

    def test_masked_nullspace_projection(self):
        """Matches the unmasked projector on the active rows; identity with
        no contact"""
        for mode in CONTACT_MODES:
            with self.subTest(contact_mode=mode):
                rows = ACTIVE_ROWS[mode]
                mask = contact_mask_from_mode(mode)
                N = np.asarray(masked_nullspace_projection(self.J, mask, self.M_inv))
                N_ref = np.asarray(nullspace_projection(self.J[rows], self.M_inv))
                np.testing.assert_allclose(N, N_ref, rtol=0, atol=1e-12)
        N = np.asarray(
            masked_nullspace_projection(self.J, contact_mask_from_mode(0), self.M_inv)
        )
        np.testing.assert_allclose(N, np.eye(self.n), rtol=0, atol=1e-15)

    def test_masked_task_space_jdot_term(self):
        """Active rows match the unmasked jdot term; masked rows are zero"""
        for mode in CONTACT_MODES:
            with self.subTest(contact_mode=mode):
                rows = ACTIVE_ROWS[mode]
                mask = np.asarray(contact_mask_from_mode(mode))
                mu = np.asarray(
                    masked_task_space_jdot_term(
                        self.qd, self.J, self.J_dot, mask, self.M_inv
                    )
                )
                mu_ref = np.asarray(
                    task_space_jdot_term(
                        self.qd, self.J[rows], self.J_dot[rows], self.M_inv
                    )
                )
                np.testing.assert_allclose(mu[rows], mu_ref, rtol=0, atol=1e-12)
                np.testing.assert_allclose(mu[mask == 0], 0.0, rtol=0, atol=1e-15)

    # -- No contact (mode 0) ----------------------------------------------------
    # There is no unmasked counterpart to compare against here (the active
    # Jacobian is empty), so the expected results are closed-form instead
    def test_no_contact_closed_forms(self):
        mask = contact_mask_from_mode(0)
        # No constraint: J_bar and the jdot term vanish entirely
        J_bar = np.asarray(
            masked_dynamically_consistent_inverse(self.J, mask, self.M_inv)
        )
        np.testing.assert_allclose(J_bar, 0.0, rtol=0, atol=1e-15)
        mu = np.asarray(
            masked_task_space_jdot_term(self.qd, self.J, self.J_dot, mask, self.M_inv)
        )
        np.testing.assert_allclose(mu, 0.0, rtol=0, atol=1e-15)
        # With N_c = identity, the underactuated projection reduces to the
        # dynamically-consistent inverse of S. (NOTE: this differs from the
        # old branched implementation, which special-cased mode 0 to S.T)
        SN_bar = np.asarray(
            masked_underactuated_contact_nullspace_projection(
                self.J, mask, self.M_inv, self.S
            )
        )
        SN_bar_ref = np.asarray(
            dynamically_consistent_inverse(np.asarray(self.S), self.M_inv)
        )
        np.testing.assert_allclose(SN_bar, SN_bar_ref, rtol=0, atol=1e-12)

    def test_masked_underactuated_contact_nullspace_projection(self):
        """Matches the unmasked underactuated projection on the active rows"""
        for mode in CONTACT_MODES:
            with self.subTest(contact_mode=mode):
                rows = ACTIVE_ROWS[mode]
                mask = contact_mask_from_mode(mode)
                SN_bar = np.asarray(
                    masked_underactuated_contact_nullspace_projection(
                        self.J, mask, self.M_inv, self.S
                    )
                )
                SN_bar_ref = np.asarray(
                    underactuated_contact_nullspace_projection(
                        self.J[rows], self.M_inv, self.S
                    )
                )
                np.testing.assert_allclose(SN_bar, SN_bar_ref, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
