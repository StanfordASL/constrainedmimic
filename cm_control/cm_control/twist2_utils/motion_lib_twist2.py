import os
import pickle
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, List
from cm_control.twist2_utils.jax_pose_utils import (
    quat_diff,
    quat_to_exp_map,
    slerp,
    euler_from_quaternion,
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
)


@jax.tree_util.register_static
class MotionLib:
    def __init__(
        self,
        motion_file,
        motion_decompose=False,
        motion_smooth=False,  # Smoothing disabled for simplicity in JAX port for now
        motion_height_adjust=False,
    ):
        self._motion_decompose = motion_decompose
        self._motion_height_adjust = motion_height_adjust

        # load motions
        self._load_motions(motion_file)

    def _load_motions(self, motion_file):
        motion_files, motion_weights = self._fetch_motion_files(motion_file)

        root_pos_list = []
        root_rot_list = []
        dof_pos_list = []
        local_body_pos_list = []
        fps_list = []
        weights_list = []

        self._body_link_list = []

        for curr_file, weight in zip(motion_files, motion_weights):
            if not os.path.exists(curr_file):
                continue
            with open(curr_file, "rb") as f:
                motion_data = pickle.load(f)

            fps = motion_data["fps"]
            root_pos = np.array(motion_data["root_pos"])
            root_rot = np.array(motion_data["root_rot"])
            dof_pos = np.array(motion_data["dof_pos"])
            local_body_pos = np.array(motion_data["local_body_pos"])

            if not self._body_link_list:
                self._body_link_list = motion_data["link_body_list"]

            if self._motion_height_adjust:
                body_pos = local_body_pos + root_pos[:, np.newaxis, :]
                lowest_body_part = np.min(body_pos[..., 2])
                root_pos[..., 2] -= lowest_body_part

            root_pos_list.append(root_pos)
            root_rot_list.append(root_rot)
            dof_pos_list.append(dof_pos)
            local_body_pos_list.append(local_body_pos)
            fps_list.append(fps)
            weights_list.append(weight)

        # Concatenate all motions into large buffers
        self._motion_num_frames = np.array([p.shape[0] for p in root_pos_list])
        self._motion_fps = np.array(fps_list)
        self._motion_dt = 1.0 / self._motion_fps
        self._motion_lengths = self._motion_dt * (self._motion_num_frames - 1)

        self._motion_weights = np.array(weights_list)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_root_pos = np.concatenate(root_pos_list, axis=0)
        self._motion_root_rot = np.concatenate(root_rot_list, axis=0)
        self._motion_dof_pos = np.concatenate(dof_pos_list, axis=0)
        self._motion_local_body_pos = np.concatenate(local_body_pos_list, axis=0)

        # Precompute derivatives
        start_indices = np.concatenate(
            [np.array([0]), np.cumsum(self._motion_num_frames)[:-1]]
        )
        self._motion_start_idx = start_indices

        # Root velocity
        root_vel = []
        for p, dt in zip(root_pos_list, fps_list):
            v = np.gradient(p, 1.0 / dt, axis=0)
            root_vel.append(v)
        self._motion_root_vel = np.concatenate(root_vel, axis=0)

        # DOF velocity
        dof_vel = []
        for p, dt in zip(dof_pos_list, fps_list):
            v = np.gradient(p, 1.0 / dt, axis=0)
            dof_vel.append(v)
        self._motion_dof_vel = np.concatenate(dof_vel, axis=0)

        # Angular velocity
        ang_vel = []
        for r, dt in zip(root_rot_list, fps_list):
            v = self._compute_so3_derivative(r, 1.0 / dt)
            ang_vel.append(v)
        self._motion_root_ang_vel = np.concatenate(ang_vel, axis=0)

        # Root pos/rot delta (local)
        root_pos_delta_local = []
        root_rot_delta_local = []
        for p, r in zip(root_pos_list, root_rot_list):
            dp = np.zeros_like(p)
            dp[1:, :] = p[1:, :] - p[:-1, :]
            dp[1:, :] = quat_rotate_inverse(r[:-1, :], dp[1:, :])
            root_pos_delta_local.append(dp)

            dr = np.zeros_like(p)
            diff = quat_diff(r[1:, :], r[:-1, :])
            dr[1:, :] = euler_from_quaternion(diff)
            dr[1:, :] = quat_rotate_inverse(r[:-1, :], dr[1:, :])
            root_rot_delta_local.append(dr)

        self._motion_root_pos_delta_local = np.concatenate(root_pos_delta_local, axis=0)
        self._motion_root_rot_delta_local = np.concatenate(root_rot_delta_local, axis=0)

        # Global root delta for looping
        self._motion_root_pos_delta = np.stack([p[-1] - p[0] for p in root_pos_list])
        self._motion_root_pos_delta[..., -1] = 0.0

        # Convert to JAX arrays for JIT compatibility
        # TODO: Confirm which of these MUST be JAX arrays vs numpy
        self._motion_num_frames = jnp.array(self._motion_num_frames)
        self._motion_lengths = jnp.array(self._motion_lengths)
        self._motion_weights = jnp.array(self._motion_weights)
        self._motion_root_pos = jnp.array(self._motion_root_pos)
        self._motion_root_rot = jnp.array(self._motion_root_rot)
        self._motion_dof_pos = jnp.array(self._motion_dof_pos)
        self._motion_local_body_pos = jnp.array(self._motion_local_body_pos)
        self._motion_start_idx = jnp.array(self._motion_start_idx)
        self._motion_root_vel = jnp.array(self._motion_root_vel)
        self._motion_dof_vel = jnp.array(self._motion_dof_vel)
        self._motion_root_ang_vel = jnp.array(self._motion_root_ang_vel)
        self._motion_root_pos_delta_local = jnp.array(self._motion_root_pos_delta_local)
        self._motion_root_rot_delta_local = jnp.array(self._motion_root_rot_delta_local)
        self._motion_root_pos_delta = jnp.array(self._motion_root_pos_delta)

    # Note: for now, this is never called in a jitted region
    # If it is, switch to using jax.numpy
    def _compute_so3_derivative(self, rotations, dt):
        if rotations.shape[0] < 3:
            root_drot = quat_diff(rotations[:-1], rotations[1:])
            omega = quat_to_exp_map(root_drot) / dt
            return np.concatenate([omega, omega[-1:]], axis=0)

        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega_interior = quat_to_exp_map(q_rel) / (2.0 * dt)

        q_start_rel = quat_mul(rotations[1], quat_conjugate(rotations[0]))
        omega_start = quat_to_exp_map(q_start_rel) / dt

        q_end_rel = quat_mul(rotations[-1], quat_conjugate(rotations[-2]))
        omega_end = quat_to_exp_map(q_end_rel) / dt

        return np.concatenate(
            [omega_start[np.newaxis], omega_interior, omega_end[np.newaxis]], axis=0
        )

    def _fetch_motion_files(self, motion_file):
        if motion_file.endswith(".yaml"):
            with open(motion_file, "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            root_path = motion_config["root_path"]
            files = [
                os.path.join(root_path, m["file"]) for m in motion_config["motions"]
            ]
            weights = [m["weight"] for m in motion_config["motions"]]
            return files, weights
        return [motion_file], [1.0]

    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]
        phase = jnp.clip(times / self._motion_lengths[motion_ids], 0.0, 1.0)

        f_idx = phase * (num_frames - 1)
        frame_idx0 = f_idx.astype(int)
        frame_idx1 = jnp.minimum(frame_idx0 + 1, num_frames - 1)
        blend = f_idx - frame_idx0

        frame_start_idx = self._motion_start_idx[motion_ids]
        return frame_idx0 + frame_start_idx, frame_idx1 + frame_start_idx, blend

    @jax.jit
    def calc_motion_frame(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        motion_loop_num = jnp.floor(motion_times / motion_len)
        curr_times = motion_times - motion_loop_num * motion_len

        idx0, idx1, blend = self._calc_frame_blend(motion_ids, curr_times)

        root_pos0, root_pos1 = self._motion_root_pos[idx0], self._motion_root_pos[idx1]
        root_rot0, root_rot1 = self._motion_root_rot[idx0], self._motion_root_rot[idx1]

        blend_exp = blend[..., jnp.newaxis]
        root_pos = (1.0 - blend_exp) * root_pos0 + blend_exp * root_pos1
        root_pos += (
            motion_loop_num[..., jnp.newaxis] * self._motion_root_pos_delta[motion_ids]
        )

        root_rot = slerp(root_rot0, root_rot1, blend)

        root_vel = self._motion_root_vel[idx0]
        root_ang_vel = self._motion_root_ang_vel[idx0]

        dof_pos = (1.0 - blend_exp) * self._motion_dof_pos[
            idx0
        ] + blend_exp * self._motion_dof_pos[idx1]
        dof_vel = self._motion_dof_vel[idx0]

        l_pos0, l_pos1 = (
            self._motion_local_body_pos[idx0],
            self._motion_local_body_pos[idx1],
        )
        local_key_body_pos = (1.0 - blend_exp[..., jnp.newaxis]) * l_pos0 + blend_exp[
            ..., jnp.newaxis
        ] * l_pos1

        dp0, dp1 = (
            self._motion_root_pos_delta_local[idx0],
            self._motion_root_pos_delta_local[idx1],
        )
        root_pos_delta_local = (1.0 - blend_exp) * dp0 + blend_exp * dp1

        dr0, dr1 = (
            self._motion_root_rot_delta_local[idx0],
            self._motion_root_rot_delta_local[idx1],
        )
        root_rot_delta_local = (1.0 - blend_exp) * dr0 + blend_exp * dr1

        return (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            local_key_body_pos,
            root_pos_delta_local,
            root_rot_delta_local,
        )

    def get_key_body_idx(self, key_body_names):
        return [self._body_link_list.index(n) for n in key_body_names]
