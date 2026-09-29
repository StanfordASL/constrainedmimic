"""
Simple state estimation with very strong assumptions

Useful for the "karate chop" hardware demo

Assume that we are standing in place with both feet
planted on the ground. Where is our pelvis and what is
its velocity in world frame?
"""

from functools import partial
from typing import Tuple

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from frax import Humanoid

from cm_control.utils.osc_utils import (
    selection_matrix,
    underactuated_contact_nullspace_projection,
)
from cm_control.utils.transform_utils import invert_transform
from cm_control.utils.rotation_utils import rmat_to_intrinsic_euler_xyz


@jax.tree_util.register_static
class FunctionalEstimator:
    def __init__(self, robot: Humanoid):
        self.robot = robot
        self.alpha_q = 0.9
        self.alpha_qd = 0.7

        # Assume that the left foot is always in contact with the world at the origin
        self.left_foot_to_world_transform = np.eye(4)
        self.S = selection_matrix(robot.num_actuated_joints)

    def q_root_from_fixed_contact_mode(self, q_actuated: Array) -> Array:
        # Note: we actually don't need to use the last estimated root state here as
        # we only need the relative transform between pelvis and foot
        q = jnp.concatenate([jnp.zeros(6), q_actuated])
        joint_transforms = self.robot.joint_to_world_transforms(q)

        # Given our current joint angles, what is the relative transform between the pelvis and feet?
        pelvis_transform = joint_transforms[5]
        left_foot_transform = self.robot._left_foot_transform(joint_transforms)

        # We assume that the left foot is planted and at the origin of the world
        world_to_foot_transform = invert_transform(left_foot_transform)
        pelvis_to_foot_transform = world_to_foot_transform @ pelvis_transform
        pelvis_to_world_transform = (
            self.left_foot_to_world_transform @ pelvis_to_foot_transform
        )
        xyz = pelvis_to_world_transform[:3, 3]
        rpy = rmat_to_intrinsic_euler_xyz(pelvis_to_world_transform[:3, :3])
        new_q_base = jnp.concatenate([xyz, rpy])
        return new_q_base

    def qd_root_from_double_support(
        self, q_root: Array, q_actuated: Array, qd_actuated: Array
    ) -> Array:
        """Estimate the root virtual joint velocities from the actuated joint
        velocities, assuming both feet remain planted on the ground
        """
        q = jnp.concatenate([q_root, q_actuated])
        joint_transforms = self.robot.joint_to_world_transforms(q)
        J_c = jnp.vstack(
            [
                self.robot._left_foot_jacobian(joint_transforms),
                self.robot._right_foot_jacobian(joint_transforms),
            ]
        )
        M = self.robot._mass_matrix(joint_transforms)
        M_inv = self.robot.mass_matrix_inverse(M)
        SN_c_bar = underactuated_contact_nullspace_projection(J_c, M_inv, self.S)
        qd = SN_c_bar @ qd_actuated
        return qd[:6]

    def step(
        self,
        new_q_act: Array,
        new_qd_act: Array,
        last_q_root: Array,
        last_qd_root: Array,
    ) -> Tuple[Array, Array]:
        new_q_root = self.q_root_from_fixed_contact_mode(new_q_act)
        new_qd_root = self.qd_root_from_double_support(
            new_q_root, new_q_act, new_qd_act
        )
        # EMA filter
        q_root = self.alpha_q * new_q_root + (1.0 - self.alpha_q) * last_q_root
        qd_root = self.alpha_qd * new_qd_root + (1.0 - self.alpha_qd) * last_qd_root
        return q_root, qd_root


class StatefulEstimator:
    def __init__(
        self,
        functional_estimator: FunctionalEstimator,
        initial_height: float = 0.790,
    ):
        self.functional_estimator = functional_estimator
        self.initial_height = initial_height

        # State management
        self.reset()

        print(f"[{type(self).__name__}]: Beginning JIT compilation")
        self._jit_compile()
        print(f"[{type(self).__name__}]: JIT compilation complete")

    def reset(self):
        """Reset the internal state estimate back to its initial value"""
        self._latest_q_root = np.array([0.0, 0.0, self.initial_height, 0.0, 0.0, 0.0])
        self._latest_qd_root = np.zeros(6)

    def _jit_compile(self):
        dummy_q_act = np.zeros(self.functional_estimator.robot.num_actuated_joints)
        jax.block_until_ready(
            self._jit_step(
                self.functional_estimator,
                dummy_q_act,
                dummy_q_act,
                self._latest_q_root,
                self._latest_qd_root,
            )
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("functional_estimator",))
    def _jit_step(functional_estimator: FunctionalEstimator, *args, **kwargs):
        return functional_estimator.step(*args, **kwargs)

    def step(self, q_act: Array, qd_act: Array) -> Tuple[Array, Array]:
        """Estimate the root state given the current actuated joint state

        Args:
            q_act (Array): Actuated joint positions, shape (29,)
            qd_act (Array): Actuated joint velocities, shape (29,)

        Returns:
            Tuple[Array, Array]:
                q_root (Array): Root virtual joint positions (position +
                    intrinsic XYZ euler angles), shape (6,)
                qd_root (Array): Root virtual joint velocities, shape (6,)
        """
        q_root, qd_root = self._jit_step(
            self.functional_estimator,
            q_act,
            qd_act,
            self._latest_q_root,
            self._latest_qd_root,
        )
        self._latest_q_root = q_root
        self._latest_qd_root = qd_root
        return q_root, qd_root
