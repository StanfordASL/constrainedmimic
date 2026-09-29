"""Utilities for constructing dynamics-level CBFs for motion tracking policies"""

import jax
from jax import Array
import jax.numpy as jnp
from cbfpy import CBF

from cm_control.core.actuators import RobotActuatorModel
from cm_control.utils.free_floating_utils import (
    mujoco_qpos_qvel_to_pose_and_twist,
    pose_and_twist_to_virtual_joints,
)


@jax.tree_util.register_static
class PDFilter:
    """Filter a PD joint target"""

    def __init__(self, cbf: CBF, actuator_model: RobotActuatorModel):
        self.cbf = cbf
        self.actuator_model = actuator_model

    @jax.jit  # TODO MOVE THIS JIT
    def filter(
        self,
        pd_target: Array,
        pos: Array,
        quat: Array,
        vel: Array,
        omega: Array,
        q_act: Array,
        qd_act: Array,
        contact_mode: int,
    ):
        q_ff, qd_ff = pose_and_twist_to_virtual_joints(pos, quat, vel, omega)
        return self.filter_from_virtual_joints(
            pd_target, q_ff, qd_ff, q_act, qd_act, contact_mode
        )

    @jax.jit  # TODO MOVE THIS JIT
    def filter_from_virtual_joints(
        self,
        pd_target: Array,
        q_ff: Array,
        qd_ff: Array,
        q_act: Array,
        qd_act: Array,
        contact_mode: int,
    ):
        z = jnp.concatenate([q_ff, q_act, qd_ff, qd_act])
        u = self.actuator_model.compute_joint_torques_from_only_positional_target(
            q_act, qd_act, pd_target
        )
        u_safe = self.cbf.safety_filter(z, u, contact_mode=contact_mode)
        return self.actuator_model.get_positional_target_from_torques(
            q_act, qd_act, u_safe
        )

    @jax.jit  # TODO MOVE THIS JIT
    def filter_from_mujoco_state(
        self, pd_target: Array, qpos: Array, qvel: Array, contact_mode: int
    ):
        pos, quat, vel, omega = mujoco_qpos_qvel_to_pose_and_twist(qpos, qvel)
        q_act = qpos[7:]
        qd_act = qvel[6:]
        return self.filter(
            pd_target, pos, quat, vel, omega, q_act, qd_act, contact_mode
        )
