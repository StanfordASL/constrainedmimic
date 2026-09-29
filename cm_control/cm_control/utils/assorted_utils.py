"""Assorted utility functions (where I'm not sure to place otherwise)"""

import jax
from jax import Array
import jax.numpy as jnp

from cm_control.utils.rotation_utils import (
    quat_wxyz_to_intrinsic_euler_xyz,
    intrinsic_euler_xyz_to_quat_wxyz,
)


# TODO make a pure numpy version of this?
def mujoco_qpos_to_q(qpos: Array) -> Array:
    return jnp.concatenate(
        [qpos[:3], quat_wxyz_to_intrinsic_euler_xyz(qpos[3:7]), qpos[7:]]
    )


# TODO make a pure numpy version of this?
def q_to_mujoco_qpos(q: Array) -> Array:
    return jnp.concatenate([q[:3], intrinsic_euler_xyz_to_quat_wxyz(q[3:6]), q[6:]])
