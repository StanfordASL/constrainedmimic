"""Utiities for loading/processing/retargeting motion data

For now, much of this assumes we are dealing specifically
with the XRobo motion data
"""

import pickle
from typing import Union
from pathlib import Path

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from cm_control.retargeting.pico_to_g1_retarget import StatefulRetargeter
from cm_control.utils.rotation_utils import (
    slerp,
    quat_wxyz_multiply,
    quat_wxyz_to_extrinsic_euler_xyz,
    rotate_vector_by_quat_wxyz,
)


def reset_pose_reference_frame(initial_pose: Array, pose: Array) -> Array:
    """Resets the reference frame of a given pose to be with respect to
    an initial pose (assuming unobservable XY position and yaw)

    Args:
        initial_pose (Array): Reference position + WXYZ quaternion pose, shape (7,)
        pose (Array): Pose to adjust, shape (7,)

    Returns:
        Array: Adjusted pose, shape (7,)
    """
    # Determine the yaw of the reference pose
    initial_yaw = quat_wxyz_to_extrinsic_euler_xyz(initial_pose[3:])[2]

    # Build the quaternion to correct for the initial yaw
    half_angle = -initial_yaw / 2.0
    q_yaw_inv = jnp.array([jnp.cos(half_angle), 0, 0, jnp.sin(half_angle)])

    # Correct the position and quaternion components for the yaw angle
    relative_position = pose[:3] - initial_pose[:3]
    corrected_pos = rotate_vector_by_quat_wxyz(q_yaw_inv, relative_position)
    corrected_quat = quat_wxyz_multiply(q_yaw_inv, pose[3:])
    return jnp.concatenate([corrected_pos, corrected_quat])


@jax.jit  # TODO: Move this JIT?
def reset_body_poses_reference_frame(initial_pose: Array, body_poses: Array) -> Array:
    """Correct a batch of poses' reference frames w.r.t an initial pose

    Args:
        initial_pose (Array): Reference position + WXYZ quaternion pose, shape (7,)
        body_poses (Array): Poses to adjust, shape (N, 7)

    Returns:
        Array: Adjusted poses, shape (N, 7)
    """
    return jax.vmap(reset_pose_reference_frame, in_axes=(None, 0))(
        initial_pose, body_poses
    )


def interpolate_poses_batch(start: Array, end: Array, pct: float) -> Array:
    """Interpolate between a starting and ending batch of poses

    Args:
        start (Array): Starting poses, shape (N, 7)
        end (Array): Ending poses, shape (N, 7)
        pct (float): Interpolation param, 0 <= pct <= 1

    Returns:
        Array: Interpolated poses, shape (N, 7)
    """
    start_pos = start[:, :3]
    end_pos = end[:, :3]
    start_quat = start[:, 3:]
    end_quat = end[:, 3:]
    # Linear interpolate position
    pos_interp = linear_interpolate(start_pos, end_pos, pct)
    # SLERP orientation
    quat_interp = slerp(start_quat, end_quat, pct)
    return jnp.hstack([pos_interp, quat_interp])


def linear_interpolate(start, end, t):
    return start + t * (end - start)


def load_xr_motion(filepath: str) -> dict:
    """Load XR motion data from a pickle file

    Args:
        filepath (str): Path to the .pkl file containing the (cropped)
            rosbag XR data

    Returns:
        dict: XR motion data with keys "times" and "poses". The time
            data has shape (num_timestep,) and the pose data has shape
            (num_timesteps, 24, 7) where 24 is the number of body poses
            returned from the XRobo interface
    """
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    return data


def clean_xr_data(data: dict, replay_freq: Union[int, float]) -> dict:
    """Convert XR data recorded at uneven timesteps to a smoothly
    interpolated version at a fixed playback frequency

    Args:
        data (dict): XR data with entries "times" and "poses"
        replay_freq (Union[int, float]): Fixed replay frequency

    Returns:
        dict: Processed XR data with entries "times" and "poses"
            and metadata "frequency"
    """
    unshifted_times = data["times"]
    # In case the time in the data starts from != 0, reset it back to 0
    times = unshifted_times - unshifted_times[0]
    poses = data["poses"]
    duration = times[-1] - times[0]
    num_samples = int(duration * replay_freq)
    replay_times = np.linspace(0.0, duration, num_samples)
    replay_data = {"times": replay_times, "poses": [], "frequency": replay_freq}
    indices = np.searchsorted(times, replay_times, side="left") - 1
    indices = np.clip(indices, 0, len(times) - 2)
    for i, t in enumerate(replay_times):
        idx = indices[i]
        idx_next = idx + 1
        t0 = times[idx]
        t1 = times[idx_next]
        dt = t1 - t0
        pct = (t - t0) / dt if dt > 0 else 0.0  # div/0 check
        pct = np.clip(pct, 0.0, 1.0)
        replay_data["poses"].append(
            interpolate_poses_batch(poses[idx], poses[idx_next], pct)
        )
    replay_data["poses"] = np.asarray(replay_data["poses"])
    return replay_data


def transform_xr_data(data: dict) -> dict:
    """Transform all poses in the data to be relative to the initial
    root pose, and reset the unobservable yaw to 0 at initialization

    Args:
        data (dict): XR data with (at minimum) "poses" info

    Returns:
        dict: Updated data with the pose information now expresed relative
            to the initial pose
    """
    poses = data["poses"]
    root_idx = 0
    initial_root_pose = poses[0][root_idx]
    transformed = []
    for body_pose in poses:
        transformed.append(
            reset_body_poses_reference_frame(initial_root_pose, body_pose)
        )
    return data | {"poses": np.asarray(transformed)}


def retarget_xr_data(data: dict, retargeter: StatefulRetargeter) -> dict:
    """Retarget XR data to the robot kinematics

    NOTE: This function assumes that we have already processed the XR data to
    maintain a smmoth, fixed frequency

    Args:
        data (dict): Processed XR data with entries "times" and "poses"
            and metadata "frequency"
        retargeter (StatefulRetargeter): Retargeter / IK solver

    Returns:
        dict: Motion data retargeted to the robot's kinematics. Contains
            entries "q", "pos", "quat", "vel", "omega", and "contact_mode",
            all defined for each timestep, and metadata "frequency"
    """
    # NOTE: we'll use a python loop here with the stateful retargeter class
    # that we use at deployment. It could be faster to write a fully functional
    # JAX version that can process an entire trajectory under JIT, but we will
    # leave this for the future.

    dt = 1 / data["frequency"]
    retargeting_data = {
        "q": [],
        "pos": [],
        "quat": [],
        "vel": [],
        "omega": [],
        "contact_mode": [],
        "frequency": data["frequency"],
    }
    for body_poses in data["poses"]:
        q, pos, quat, vel, omega, contact_mode = retargeter.step(body_poses, dt)
        retargeting_data["q"].append(q)
        retargeting_data["pos"].append(pos)
        retargeting_data["quat"].append(quat)
        retargeting_data["vel"].append(vel)
        retargeting_data["omega"].append(omega)
        retargeting_data["contact_mode"].append(contact_mode)

    # Convert to numpy and return
    for k, v in retargeting_data.items():
        if isinstance(v, list):
            retargeting_data[k] = np.asarray(v)
    return retargeting_data


def load_and_retarget_motion(
    filepath: Union[str, Path],
    replay_freq: Union[int, float],
    retargeter: StatefulRetargeter,
) -> dict:
    """Load an XR motion file and retarget it to the robot kinematics

    Args:
        filepath (Union[str, Path]): Path to the .pkl file containing the (cropped)
            rosbag XR data
        replay_freq (Union[int, float]): Fixed replay frequency
        retargeter (StatefulRetargeter): Retargeter / IK solver

    Returns:
        dict: Motion data retargeted to the robot's kinematics. Contains
            entries "q", "pos", "quat", "vel", "omega", and "contact_mode",
            all defined for each timestep, and metadata "frequency"
    """
    motion_data = load_xr_motion(filepath)
    motion_data = clean_xr_data(motion_data, replay_freq)
    motion_data = transform_xr_data(motion_data)
    print(
        f"[{Path(__file__).name}]: XR data loaded and processed. Starting retargeting."
    )
    retargeting_data = retarget_xr_data(motion_data, retargeter)
    print(f"[{Path(__file__).name}]: Retargeting complete.")
    return retargeting_data
