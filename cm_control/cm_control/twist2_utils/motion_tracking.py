"""RL-based whole-body motion tracking, based on twist2"""

from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from cm_control.twist2_utils import twist2_config
from cm_control.twist2_utils.jax_twist2 import load_policy
from cm_control.twist2_utils.twist2_utils import (
    build_mimic_obs_from_world_velocities,
    build_full_obs_and_new_history,
    build_proprio_obs,
)


@jax.tree_util.register_static
class FunctionalMotionTracker:
    """Static, functional JAX whole-body tracking"""

    def __init__(self):
        self.nn_model, self.nn_params = load_policy()
        self.num_actions = 29
        self.action_scale = twist2_config.action_scale
        self.default_dof_pos = np.asarray(twist2_config.default_joint_position)
        self.n_mimic_obs = 35
        self.history_len = 10
        self.n_obs_single = 127

    def step(
        self,
        robot_quat_wxyz: Array,
        robot_omega_body: Array,
        robot_q_actuated: Array,
        robot_qd_actuated: Array,
        mimic_pos: Array,
        mimic_quat: Array,
        mimic_vel: Array,
        mimic_omega: Array,
        mimic_q_act: Array,
        history: Array,
        last_action: Array,
    ):
        mimic_obs = build_mimic_obs_from_world_velocities(
            mimic_pos,
            mimic_quat,
            mimic_vel,
            mimic_omega,
            mimic_q_act,
        )
        proprio_obs = build_proprio_obs(
            robot_quat_wxyz,
            robot_omega_body,
            robot_q_actuated,
            robot_qd_actuated,
            last_action,
        )
        full_obs, new_history = build_full_obs_and_new_history(
            mimic_obs, proprio_obs, history
        )
        raw_action = self.nn_model.apply(
            self.nn_params, full_obs[jnp.newaxis, :], train=False
        ).squeeze()
        pd_target = raw_action * self.action_scale + self.default_dof_pos
        return pd_target, raw_action, new_history


class StatefulMotionTracker:
    """Stateful wrapper around purely functional JAX policy and observation logic"""

    def __init__(self, functional_tracker: FunctionalMotionTracker):
        self.functional_tracker = functional_tracker

        # States
        # TODO decide if more data shold be cached in the state?
        self._last_action = np.zeros(functional_tracker.num_actions)
        self._obs_history = np.zeros(
            (functional_tracker.history_len, functional_tracker.n_obs_single)
        )

        # Jit compile the JAX logic
        print(f"[{type(self).__name__}]: Beginning JIT compilation")
        self._jit_compile()
        print(f"[{type(self).__name__}]: JIT compilation complete")

    def _update_state(self, action: Array, obs_history: Array) -> None:
        self._last_action = action
        self._obs_history = obs_history

    def _jit_compile(self):
        dummy_vec3 = np.zeros(3)
        dummy_quat = np.array([1.0, 0.0, 0.0, 0.0])
        dummy_q_act = np.zeros(29)
        dummy_proprio_args = (dummy_quat, dummy_vec3, dummy_q_act, dummy_q_act)
        dummy_mimic_args = (dummy_vec3, dummy_quat, dummy_vec3, dummy_vec3, dummy_q_act)
        jax.block_until_ready(
            self._jit_step(
                self.functional_tracker,
                *dummy_proprio_args,
                *dummy_mimic_args,
                self._obs_history,
                self._last_action,
            )
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("functional_tracker",))
    def _jit_step(functional_tracker: FunctionalMotionTracker, *args, **kwargs):
        return functional_tracker.step(*args, **kwargs)

    # TODO make a more minimal version of this that only takes in the
    # proprioceptive/mimc info that is used to construct things?
    def step(
        self,
        robot_quat_wxyz: Array,
        robot_omega_body: Array,
        robot_q: Array,
        robot_qd: Array,
        mimic_pos: Array,
        mimic_quat: Array,
        mimic_vel: Array,
        mimic_omega: Array,
        mimic_q: Array,
    ) -> Array:
        """Evaluate the motion tracking policy

        Args:
            robot_quat_wxyz (Array): Robot WXYZ quaternion, shape (4,)
            robot_omega_body (Array): Robot angular velocity (body frame), shape (3,)
            robot_q (Array): Robot actuated joint positions, shape (29,)
            robot_qd (Array): Robot actuated joint velocities, shape (29,)
            mimic_pos (Array): Mimic root position, shape (3,)
            mimic_quat (Array): Mimic WXYZ quaternion, shape (4,)
            mimic_vel (Array): Mimic root velocity, shape (3,)
            mimic_omega (Array): Mimic root angular velocity (world frame), shape (3,)
            mimic_q (Array): Mimic actuated joint positions, shape (29,)

        Returns:
            Array: PD joint target, shape (29,)
        """
        proprio_args = (robot_quat_wxyz, robot_omega_body, robot_q, robot_qd)
        mimic_args = (mimic_pos, mimic_quat, mimic_vel, mimic_omega, mimic_q)
        pd_target, raw_action, new_history = self._jit_step(
            self.functional_tracker,
            *proprio_args,
            *mimic_args,
            self._obs_history,
            self._last_action,
        )
        self._update_state(raw_action, new_history)
        return pd_target
