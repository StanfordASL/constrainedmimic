"""Utility functions for anything related to the TWIST2 controller"""

from typing import Optional, Tuple

from jax import Array
import jax.numpy as jnp

from cm_control.config import g1_config
from cm_control.twist2_utils import twist2_config
from cm_control.twist2_utils.motion_lib_twist2 import MotionLib
from cm_control.utils.rotation_utils import (
    quat_wxyz_to_extrinsic_euler_xyz,
    rotate_vector_by_quat_wxyz,
    quat_wxyz_conjugate,
)


def build_mimic_obs(
    mimic_pos: Array,
    mimic_quat_wxyz: Array,
    mimic_vel_body: Array,
    mimic_omega_body: Array,
    mimic_q: Array,
) -> Array:
    """Build the mimic observation vector for TWIST2

    Args:
        mimic_pos (Array): Position of the root (pelvis), shape (3,).
            Note that only the Z component is used.
        mimic_quat_wxyz (Array): Orientation of the root (pelvis) as a WXYZ quaternion, shape (4,)
        mimic_vel_body (Array): Velocity of the root (pelvis) in body frame, shape (3,).
            Note that only the X/Y velocity is used.
        mimic_omega_body (Array): Angular velocity of the root (pelvis) in body frame, shape (3,).
            Note that only the Z component (yaw rate) is used.
        mimic_q (Array): Positions of all actuated joints, shape (29,)

    Returns:
        Array: Observation vector, shape (35,)
    """
    assert mimic_pos.shape == (3,)
    assert mimic_quat_wxyz.shape == (4,)
    assert mimic_vel_body.shape == (3,)
    assert mimic_omega_body.shape == (3,)
    assert mimic_q.shape == (29,)
    rpy = quat_wxyz_to_extrinsic_euler_xyz(mimic_quat_wxyz)
    return jnp.concatenate(
        [
            mimic_vel_body[:2],  # X/Y velocity in body frame
            jnp.atleast_1d(mimic_pos[2]),  # Root z height
            rpy[:2],  # Roll and pitch
            jnp.atleast_1d(mimic_omega_body[2]),  # Yaw angular velocity in body frame
            mimic_q,  # 29-DOF joint angles
        ]
    )


def build_mimic_obs_from_world_velocities(
    mimic_pos: Array,
    mimic_quat_wxyz: Array,
    mimic_vel_world: Array,
    mimic_omega_world: Array,
    mimic_q: Array,
) -> Array:
    """Constructs the mimic obs, assuming the mimic velocities are provided in WORLD frame

    Args:
        mimic_pos (Array): Position of the root (pelvis), shape (3,).
            Note that only the Z component is used.
        mimic_quat_wxyz (Array): Orientation of the root (pelvis) as a WXYZ quaternion, shape (4,)
        mimic_vel_world (Array): Velocity of the root (pelvis) in WORLD frame, shape (3,)
        mimic_omega_world (Array): Angular velocity of the root (pelvis) in WORLD frame, shape (3,)
        mimic_q (Array): Positions of all actuated joints, shape (29,)

    Returns:
        Array: Observation vector, shape (35,)
    """
    quat_conj = quat_wxyz_conjugate(mimic_quat_wxyz)
    vel_body = rotate_vector_by_quat_wxyz(quat_conj, mimic_vel_world)
    omega_body = rotate_vector_by_quat_wxyz(quat_conj, mimic_omega_world)
    return build_mimic_obs(mimic_pos, mimic_quat_wxyz, vel_body, omega_body, mimic_q)


def build_proprio_obs(
    proprio_quat_wxyz: Array,
    proprio_omega_body: Array,
    proprio_q: Array,
    proprio_qdot: Array,
    last_action: Array,
) -> Array:
    """Build the proprioception observation vector for the TWIST2 policy

    Args:
        proprio_quat_wxyz (Array): Orientation of the root (pelvis) as a WXYZ quaternion, shape (4,)
        proprio_omega_body (Array): Angular velocity of the root (pelvis) in body frame, shape (3,)
        proprio_q (Array): Positions of all actuated joints, shape (29,)
        proprio_qdot (Array): Velocities of all actuated joints, shape (29,)
        last_action (Array): Previous action from the policy, shape (29,).
            Note that this is *not* directly the joint PD targets

    Returns:
        Array: Observation vector, shape (92,)
    """
    assert proprio_quat_wxyz.shape == (4,)
    assert proprio_omega_body.shape == (3,)
    assert proprio_q.shape == (29,)
    assert proprio_qdot.shape == (29,)
    assert last_action.shape == (29,)
    rpy = quat_wxyz_to_extrinsic_euler_xyz(proprio_quat_wxyz)
    # Zero out ankle joint velocities
    proprio_qdot = proprio_qdot.at[jnp.asarray(g1_config.ankle_indices)].set(0.0)
    return jnp.concatenate(
        [
            proprio_omega_body * twist2_config.ang_vel_scale,
            rpy[:2],
            (proprio_q - jnp.asarray(twist2_config.default_joint_position))
            * twist2_config.dof_pos_scale,
            proprio_qdot * twist2_config.dof_vel_scale,
            last_action,
        ]
    )


def build_full_obs_and_new_history(
    mimic_obs: Array,
    proprio_obs: Array,
    obs_history: Array,
    future_obs: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Builds the full observation vector (input to twist2 policy) and returns the new history

    Args:
        mimic_obs (Array): Mimic observation vector, shape (35,)
        proprio_obs (Array): Proprioceptive observation vector, shape (92,)
        obs_history (Array): History of last 10 mimic/proprioceptive observations, shape (10, 127)
        future_obs (Optional[Array]): Future observation vector, used for autonomous rollouts.
            Defaults to None, as used in teleoperation. If not None, shape (35,)

    Returns:
        Tuple[Array, Array]:
            full_obs (Array): Full observation vector to pass to the policy, shape (1432,).
            new_history (Array): New history with the latest observations, shape (10, 127).
    """
    assert mimic_obs.shape == (35,)
    assert proprio_obs.shape == (92,)
    assert obs_history.shape == (10, 127)
    if future_obs is None:
        # During teleoperation, no future obs -- set to mimic as placeholder
        future_obs = mimic_obs
    else:
        assert future_obs.shape == mimic_obs.shape
    mimic_and_proprio_obs = jnp.concatenate([mimic_obs, proprio_obs])
    flat_obs_history = obs_history.flatten()
    full_obs = jnp.concatenate([mimic_and_proprio_obs, flat_obs_history, future_obs])
    new_history = jnp.roll(obs_history, shift=-1, axis=0)
    new_history = new_history.at[-1].set(mimic_and_proprio_obs)
    return full_obs, new_history


# TODO: add documentation and figure out the meaning/shape of tar_motion_steps
# and maybe simplify things a bit
def build_mimic_obs_from_motion_lib(
    motion_lib: MotionLib, t_step: int, control_dt: float, tar_motion_steps: Array
):
    motion_times = jnp.expand_dims(t_step * control_dt, axis=-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()

    # Assuming motion_ids is always 0 for single trajectory
    motion_ids = jnp.zeros(len(tar_motion_steps), dtype=int)

    # WARNING: all of the motion_lib code assumes XYZW quaternions
    # Everywhere else we use WXYZ
    (
        root_pos,
        root_quat_xyzw,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        local_key_body_pos,
        root_pos_delta_local,
        root_rot_delta_local,
    ) = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Convert to WXYZ, and flatten the output of motion_lib, which includes a batch dim = 1
    root_quat_wxyz = root_quat_xyzw.squeeze()[jnp.array([3, 0, 1, 2])]
    return build_mimic_obs_from_world_velocities(
        root_pos.squeeze(),
        root_quat_wxyz,
        root_vel.squeeze(),
        root_ang_vel.squeeze(),
        dof_pos.squeeze(),
    )
