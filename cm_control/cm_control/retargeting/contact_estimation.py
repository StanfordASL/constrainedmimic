"""Heuristic contact mode estimation

Note: we assume that there is always at least one foot on the ground
(contact mode = 1, 2, or 3). During (safe) teleoperation, this should
always be the case.

This estimator will also likely be called from a *stateful* wrapper
class which keeps track of the latest contact mode. Note that this
implies that we will need to be smart about what in the class is
marked as static for jit
"""

from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from cm_control.utils.jax_utils import is_scalar_float, is_scalar_int


# TODO: Add check for if it seems like we are transitioning from mode 1->2 without 3 in between
# Under our "one foot on the ground assumption" this should not be possible


# @partial(
#     jax.jit,
#     static_argnames=(
#         "height_threshold",
#         "normal_vel_threshold",
#         "planar_vel_threshold",
#         "min_contact_frames",
#     ),
# )
def contact_mode_heuristic(
    # Height
    left_foot_z: float,
    right_foot_z: float,
    height_threshold: float,
    # Velocity
    left_foot_vel: Optional[Array],
    right_foot_vel: Optional[Array],
    normal_vel_threshold: Optional[float],
    planar_vel_threshold: Optional[float],
    # Hysteresis
    prev_contact_mode: Optional[int],
    num_contact_frames: Optional[int],
    min_contact_frames: Optional[int],
) -> int:
    """Contact mode estimator

    We determine the contact mode based on a few conditions:
    1. Foot height -- if low to the ground, likely in contact
    2. Foot velocity (planar and normal) -- if stopped, likely in contact
    3. Previous contact mode history -- don't switch contact modes too fast

    Velocity and previous contact mode history data are optional inputs that
    will improve the estimate if it is available. If not, set these to None

    Args:
        left_foot_z (float): Left foot z position
        right_foot_z (float): Right foot z position
        height_threshold (float): Threshold: lower than this value implies contact
        left_foot_vel (Optional[Array]): Left foot velocity, shape (3,)
        right_foot_vel (Optional[Array]): Right foot velocity, shape (3,)
        normal_vel_threshold (Optional[float]): Threshold: velocity magnitudes
            (in the normal direction) below this value implies contact
        planar_vel_threshold (Optional[float]): Threshold: velocity magnitudes
            (in the planar direction) below this value implies contact
        prev_contact_mode (Optional[int]): Previous contact mode
        num_contact_frames (Optional[int]): Number of times that the robot has
            been in prev_contact_mode
        min_contact_frames (Optional[int]): Minimum number of times that the robot
            has been in prev_contact_mode to allow for switching contact modes

    Returns:
        int: Estimated contact mode. 1 = left foot, 2 = right foot, 3 = both feet.
            Mode 0 (no contact) is not handled, for now
    """
    # Check inputs + static attribs
    assert is_scalar_float(left_foot_z)
    assert is_scalar_float(right_foot_z)
    assert is_scalar_float(height_threshold)
    vel_args = (
        left_foot_vel,
        right_foot_vel,
        normal_vel_threshold,
        planar_vel_threshold,
    )
    has_velocity_data = any(arg is not None for arg in vel_args)
    if has_velocity_data:
        assert left_foot_vel.shape == (3,)
        assert right_foot_vel.shape == (3,)
        assert is_scalar_float(normal_vel_threshold)
        assert is_scalar_float(planar_vel_threshold)
    hysteresis_args = (prev_contact_mode, num_contact_frames, min_contact_frames)
    has_hysteresis = any(arg is not None for arg in hysteresis_args)
    if has_hysteresis:
        assert is_scalar_int(prev_contact_mode)
        assert is_scalar_int(num_contact_frames)
        assert is_scalar_int(min_contact_frames)

    # CONDITION 1: height-based
    lowest_foot_z = jnp.minimum(left_foot_z, right_foot_z)
    left_is_near_ground = (left_foot_z - lowest_foot_z) < height_threshold
    right_is_near_ground = (right_foot_z - lowest_foot_z) < height_threshold

    # CONDITION 2: velocity-based (if provided)
    if not has_velocity_data:
        left_in_contact = left_is_near_ground
        right_in_contact = right_is_near_ground
    else:
        left_normal_vel = jnp.abs(left_foot_vel[2])
        right_normal_vel = jnp.abs(right_foot_vel[2])
        left_planar_vel = jnp.linalg.norm(left_foot_vel[:2])
        right_planar_vel = jnp.linalg.norm(right_foot_vel[:2])

        left_is_stopped = (left_normal_vel < normal_vel_threshold) & (
            left_planar_vel < planar_vel_threshold
        )
        right_is_stopped = (right_normal_vel < normal_vel_threshold) & (
            right_planar_vel < planar_vel_threshold
        )
        left_in_contact = left_is_near_ground & left_is_stopped
        right_in_contact = right_is_near_ground & right_is_stopped

    # Fallback in case case neither foot is detected to be in contact
    # (This probably won't happen in practice)
    if has_hysteresis:
        fallback_contact_mode = prev_contact_mode
    else:
        fallback_contact_mode = 3  # Both feet

    candidate_contact_mode = jnp.where(
        left_in_contact,
        jnp.where(right_in_contact, 3, 1),
        jnp.where(right_in_contact, 2, fallback_contact_mode),
    )

    # CONDITION 3: hysteresis (if provided)
    if not has_hysteresis:
        return candidate_contact_mode
    # Only transition between modes after some duration in a given mode,
    # to suppress high-frequency noisy switching
    return jnp.where(
        num_contact_frames >= min_contact_frames,
        candidate_contact_mode,
        prev_contact_mode,
    )
