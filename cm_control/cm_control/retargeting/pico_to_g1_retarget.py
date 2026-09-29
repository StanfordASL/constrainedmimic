"""Retargeting from the Pico VR data to the Unitree G1"""

from typing import Union, Tuple, Optional, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from cbfpy import CBF
import numpy as np

from cm_control.config.retargeting_config import (
    RETARGET_KP_POS,
    RETARGET_KP_ROT,
    RETARGET_WEIGHT_POS,
    RETARGET_WEIGHT_ROT,
    G1_RETARGETING_IDXS,
    PICO_RETARGETING_IDXS,
    PICO_TO_G1_SCALING_FACTORS,
)
from cm_control.config import g1_config
from cm_control.twist2_utils import twist2_config
from frax import Humanoid
from cm_control.retargeting.retarget import KinematicRetargeter
from cm_control.retargeting.contact_estimation import contact_mode_heuristic
from cm_control.utils.jax_utils import is_scalar_float, is_scalar_int
from cm_control.utils.filtering_utils import pose_filter
from cm_control.utils.rotation_utils import (
    intrinsic_euler_xyz_to_quat_wxyz,
    quat_wxyz_to_intrinsic_euler_xyz,
    quat_wxyz_to_rmat,
    Rz,
)


class RetargeterState(NamedTuple):
    q: Array
    root_pos: Array
    root_quat_wxyz: Array
    root_vel: Array
    root_omega: Array
    feet_pos: Array
    contact_mode: int
    num_contact_frames: int

    @classmethod
    def initial(cls) -> "RetargeterState":
        """Construct a new state with all initial values"""
        # Height is calibrated to be just in contact at the default joint pos
        # TODO: Note that this differs from the 0.8 value in twist2
        default_root_pos = jnp.array([0, 0, 0.779])
        default_root_euler = jnp.zeros(3)
        default_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        default_vel = jnp.zeros(3)
        q_init = jnp.concatenate(
            [
                default_root_pos,
                default_root_euler,
                twist2_config.default_joint_position,
            ]
        )
        # Foot values are calibrated for the above q_init and my G1 model
        default_foot_pos = jnp.array(
            [
                [0.04226619, 0.11850645, 0.00482552],
                [0.04226619, -0.11850645, 0.00482552],
            ]
        )
        return cls(
            q=q_init,
            root_pos=default_root_pos,
            root_quat_wxyz=default_quat,
            root_vel=default_vel,
            root_omega=default_vel,
            feet_pos=default_foot_pos,
            contact_mode=3,
            num_contact_frames=0,
        )

    def check(self) -> None:
        assert self.q.shape == (35,)
        assert self.root_pos.shape == (3,)
        assert self.root_quat_wxyz.shape == (4,)
        assert self.root_vel.shape == (3,)
        assert self.root_omega.shape == (3,)
        assert self.feet_pos.shape == (2, 3)
        assert is_scalar_int(self.contact_mode)
        assert is_scalar_int(self.num_contact_frames)


@jax.tree_util.register_static
class PicoToG1Retargeter:
    """Kinematic retargeter from Pico XR data -> Unitree G1

    Args:
        cbf (Optional[CBF]): CBF configuration for constrained kinematic retargeting.
            Set to None if no CBF is desired.
        robot (Humanoid): Humanoid to retarget (i.e. Unitree g1). Currently,
            we assume that we are retargeting a floating-base humanoid
        use_constrained_integration (bool, optional): Whether to integrate based on the
            contact-constrained kinematics. Defaults to False (unconstrained)
        force_double_support (bool, optional): Force contact mode 3 (both feet) on
            every online step. Defaults to False.
        stepsize (float): Integration stepsize. 1.0 = fast convergence, less than 1.0
            is slower to converge but more stable. Defaults to 1.0
        convergence_tol (float): Convergence tolerance: retargeting converged if
            the maximum difference between consecutive IK joint angles < tol.
            Defaults to 1e-3.
        max_iters (int): Maximum number of iterations for the inner differential IK loop.
            Defaults to 50.
        regularization (Optional[float]): Tikhonov regularization. Defaults to 1e-12.
        solver_tol (float, optional): Tolerance for the differential IK QP solver. Defaults to 1e-5.
    """

    def __init__(
        self,
        cbf: Optional[CBF],
        robot: Humanoid,
        use_constrained_integration: bool = False,  # TODO experiment more
        force_double_support: bool = False,  # TODO experiment more
        stepsize: float = 1.0,
        convergence_tol: float = 1e-3,
        max_iters: int = 50,
        regularization: Optional[float] = 1e-12,
        solver_tol: float = 1e-5,
    ):

        # Other fixed values
        # TODO decide if moving to config or if config should move here
        self.contact_z_tol = 0.01
        self.foot_com_to_ground_distance = g1_config.foot_com_to_ground_distance
        # EMA smoothing parameters
        self.alpha_vel = 0.7
        self.alpha_omega = 0.7
        # Indices of the pico pose data that we want to use for retargeting
        self.pico_idxs = np.asarray(PICO_RETARGETING_IDXS)
        # Indices of the feet in the *selected for retargeting* pico idxs
        # The left foot and right foot indices of the pico are 10 and 11
        # BUT in the once we've indexed the links we care about, the new
        # indices we need to select are 3 and 6
        self.feet_idxs = np.array([3, 6])
        # Scaling factors for the (selected) pico pose data to roughly match the G1 form factor
        self.scaling_factors = np.asarray(PICO_TO_G1_SCALING_FACTORS)
        if self.scaling_factors.ndim == 1:
            self.scaling_factors = self.scaling_factors[:, np.newaxis]
        # Index corresponding to the root (pelvis) in the pico data
        self.root_index_pico = 0

        # Parameters for contact estimation
        self.contact_est_height_thresh = 0.01  # TODO TUNE AND MOVE THIS
        self.contact_est_normal_vel_thresh = 0.01  # TODO TUNE AND MOVE THIS
        self.contact_est_planar_vel_thresh = 0.01  # TODO TUNE AND MOVE THIS
        self.contact_est_min_frames = 3  # TODO TUNE AND MOVE THIS

        self.use_constrained_integration = use_constrained_integration
        self.force_double_support = force_double_support

        # The general task-space IK engine, held by composition
        self.ik = KinematicRetargeter(
            cbf,
            robot,
            np.asarray(G1_RETARGETING_IDXS),
            RETARGET_KP_POS,
            RETARGET_KP_ROT,
            RETARGET_WEIGHT_POS,
            RETARGET_WEIGHT_ROT,
            stepsize,
            convergence_tol,
            max_iters,
            regularization,
            solver_tol,
        )

    # TODO: better handling of world/local reference frame
    # Right now, the positions are defined in world, which should be ok
    # as long as there isn't uneven scaling between x and y
    def apply_scaling(self, positions: Array) -> Array:
        """Scale the positions of the XRobo human pose data to better match the robot size

        Args:
            positions (Array): Positions of all frames used for XRobo -> robot retargeting, shape (n, 3)

        Returns:
            Array: Rescaled position data, shape (n, 3)
        """
        positions = jnp.atleast_2d(positions)
        num_frames = positions.shape[0]
        assert positions.shape == (num_frames, 3)
        root_scales = self.scaling_factors[self.root_index_pico]
        root_position = positions[self.root_index_pico]
        unscaled_deltas = positions - root_position
        scaled_root = root_scales * root_position
        scaled_deltas = self.scaling_factors * unscaled_deltas
        updated_positions = scaled_root + scaled_deltas
        return updated_positions

    def update_foot_rotations(self, target_rmats: Array) -> Array:
        """Before retargeting, make sure that the orientations of the feet are
        parallel with the ground, to maintain planar contact

        Args:
            target_rmats (Array): Pre-retargeting rotation matrices from the reference body,
                shape (n, 3, 3)

        Returns:
            Array: Adjusted set of rotation matrices, shape (n, 3, 3)
        """
        foot_rmats = target_rmats[self.feet_idxs]
        # Extract yaw from each matrix
        foot_yaws = jnp.atan2(foot_rmats[:, 1, 0], foot_rmats[:, 0, 0])
        # Update the rotation matrices to match the yaw but stay parallel to the ground
        new_foot_rmats = jax.vmap(Rz)(foot_yaws)
        return target_rmats.at[self.feet_idxs].set(new_foot_rmats)

    def update_heights(self, target_pos: Array) -> Array:
        # Determine the lowest height across both the left and right feet
        lowest_foot_z = jnp.minimum(
            target_pos[self.feet_idxs[0], 2], target_pos[self.feet_idxs[1], 2]
        )
        # Update the height of all bodies to make sure that one foot is planted on the ground
        z_offset = -1 * lowest_foot_z + self.foot_com_to_ground_distance
        return target_pos + jnp.array([[0, 0, z_offset]])

    def postprocess_retargeting(
        self, q_next: Array, state: RetargeterState, dt: float
    ) -> Tuple[Array, Array, Array, Array]:
        # Convert our representation of the generalized coordinates q to joint angles
        # and the position and quaternion of the floating root (pelvis)
        pos = q_next[:3]
        euler = q_next[3:6]
        quat_wxyz = intrinsic_euler_xyz_to_quat_wxyz(euler)

        # Apply EMA smoothing to the root twist
        # Note: we don't filter the pose for now. If we decide to do so
        # in the future, would need to also update the freefloating base's
        # DOFs in q
        pos, quat, filtered_vel, filtered_omega = pose_filter(
            pos=pos,
            quat=quat_wxyz,
            vel=None,  # Infer from delta pos
            omega=None,  # Infer from delta quat
            last_pos=state.root_pos,
            last_quat=state.root_quat_wxyz,
            last_vel=state.root_vel,
            last_omega=state.root_omega,
            alpha_pos=None,  # No filtering for now
            alpha_quat=None,  # No filtering for now
            alpha_vel=self.alpha_vel,
            alpha_omega=self.alpha_omega,
            dt=dt,
        )
        # TODO decide if we should even return the position and orientation here?
        return pos, quat, filtered_vel, filtered_omega

    def step(
        self, new_body_poses: Array, state: RetargeterState, dt: float
    ) -> RetargeterState:
        """Online retargeting step: pure function of (poses, state, dt).

        This is the deployment-time step. It optionally uses contact-constrained
        integration and can force double support (see the class-level flags).
        """
        return self._step(
            new_body_poses,
            state,
            dt,
            use_constrained_integration=self.use_constrained_integration,
            force_double_support=self.force_double_support,
        )

    def scp_initialization_step(
        self, new_body_poses: Array, state: RetargeterState, dt: float
    ) -> RetargeterState:
        """Single SCP initialization step: pure function of (poses, state, dt).

        Differs from ``step`` only in that it never uses constrained integration
        and never forces double support -- it is used to converge the robot onto
        the very first datapoint from an arbitrary initial pose.
        """
        return self._step(
            new_body_poses,
            state,
            dt,
            use_constrained_integration=False,
            force_double_support=False,
        )

    def _step(
        self,
        new_body_poses: Array,
        state: RetargeterState,
        dt: float,
        use_constrained_integration: bool,
        force_double_support: bool,
    ) -> RetargeterState:
        """Shared implementation for the online and SCP-initialization steps.

        The two callers differ only in the two boolean flags, which are static
        (Python bools) so the JIT specializes cleanly on each.
        """
        # Check inputs
        assert new_body_poses.shape == (24, 7)
        assert is_scalar_float(dt)
        state.check()

        # Preprocessing: Deal with inputs BEFORE the diff IK step
        target_pos, target_rmats, new_feet_pos, contact_mode = self._preprocess_step(
            new_body_poses, state, dt
        )
        if force_double_support:  # ONLY FOR ONLINE TESTING
            contact_mode = 3
        # Run diff IK
        q_dot = self.ik.diff_ik_step(
            state.q, target_pos, target_rmats, contact_mode=contact_mode
        )
        # Postprocessing: Deal with the outputs AFTER the diff IK step
        if use_constrained_integration:
            q_next = self.ik.integrate_qdot(
                state.q, q_dot, dt, contact_mode=contact_mode
            )
        else:
            q_next = self.ik.integrate_qdot(state.q, q_dot, dt, contact_mode=None)
        pos, quat_wxyz, filtered_vel, filtered_omega = self.postprocess_retargeting(
            q_next, state, dt
        )
        # Track how long we have been in the current contact mode (hysteresis)
        num_contact_frames = jnp.where(
            contact_mode == state.contact_mode, state.num_contact_frames + 1, 0
        )
        return RetargeterState(
            q=q_next,
            root_pos=pos,
            root_quat_wxyz=quat_wxyz,
            root_vel=filtered_vel,
            root_omega=filtered_omega,
            feet_pos=new_feet_pos,
            contact_mode=contact_mode,
            num_contact_frames=num_contact_frames,
        )

    def _preprocess_step(
        self, new_body_poses: Array, state: RetargeterState, dt: float
    ):
        # Preprocessing: Deal with inputs BEFORE the diff IK step

        ### HANDLE PICO DATA ###

        all_positions = new_body_poses[:, :3]
        all_quats = new_body_poses[:, 3:]
        # Select just the pose information required for retargeting
        target_pos = all_positions[self.pico_idxs]
        target_quats = all_quats[self.pico_idxs]
        target_rmats = jax.vmap(quat_wxyz_to_rmat)(target_quats)
        # Flatten the feet to avoid non-planar contacts
        target_rmats = self.update_foot_rotations(target_rmats)
        # Then, re-scale the positional data based on the human -> G1 size difference
        target_pos = self.apply_scaling(target_pos)
        # Then, update the height of all bodies based on the height of the lowest body
        target_pos = self.update_heights(target_pos)

        ### HANDLE FEET DATA ###

        new_feet_pos = target_pos[self.feet_idxs]
        # TODO: Decide if an EMA filter needs to be here on the estimate of the foot velocity
        # I do already apply an EMA filter on the pico data in the xrobo node
        # So, perhaps the position data is smooth enough to not need this
        feet_vels = (new_feet_pos - state.feet_pos) / dt
        contact_mode = contact_mode_heuristic(
            left_foot_z=new_feet_pos[0, 2],
            right_foot_z=new_feet_pos[1, 2],
            height_threshold=self.contact_est_height_thresh,
            left_foot_vel=feet_vels[0],
            right_foot_vel=feet_vels[1],
            normal_vel_threshold=self.contact_est_normal_vel_thresh,
            planar_vel_threshold=self.contact_est_planar_vel_thresh,
            prev_contact_mode=state.contact_mode,
            num_contact_frames=state.num_contact_frames,
            min_contact_frames=self.contact_est_min_frames,
        )

        return target_pos, target_rmats, new_feet_pos, contact_mode


class StatefulRetargeter:
    """Stateful wrapper around the functional core retargeting logic

    This will purely do state management across multiple timesteps,
    and call static, JIT-compiled functions that take in this state data
    """

    def __init__(self, functional_retargeter: PicoToG1Retargeter):
        self.functional_retargeter = functional_retargeter

        # All per-timestep state now lives in a single PyTree
        self.state = RetargeterState.initial()

        # Jit compile the JAX logic
        print(f"[{type(self).__name__}]: Beginning JIT compilation")
        self._jit_compile()
        print(f"[{type(self).__name__}]: JIT compilation complete")

        # Track if this is the first time we've received data
        # If so, we need to run an initial SCP step to align the
        # robot state with the initial data
        self._is_initialized = False

    @staticmethod
    @partial(jax.jit, static_argnames=("retargeter"))
    def _jit_step(retargeter: PicoToG1Retargeter, *args, **kwargs):
        return retargeter.step(*args, **kwargs)

    # TODO rewrite this to use the full jit throught the while loop
    @staticmethod
    @partial(jax.jit, static_argnames=("retargeter"))
    def _jit_scp_step(retargeter: PicoToG1Retargeter, *args, **kwargs):
        return retargeter.scp_initialization_step(*args, **kwargs)

    def _jit_compile(self):
        dummy_poses = np.zeros((24, 7))
        dummy_dt = 0.01
        args = (self.functional_retargeter, dummy_poses, self.state, dummy_dt)
        jax.block_until_ready(self._jit_step(*args))
        jax.block_until_ready(self._jit_scp_step(*args))

    # Inputs are purely the new data which changes on a per-timestep basis
    def step(self, new_body_poses: Array, dt: float):
        if not self._is_initialized:
            return self.scp_initialization(new_body_poses)
        self.state = self._jit_step(
            self.functional_retargeter, new_body_poses, self.state, dt
        )
        return self._outputs()

    def scp_initialization(self, initial_body_poses: Array):

        # Correct for initial global error -- reset the initial pose
        # of the robot's pelvis to match the first datapoint
        root_idx = 0
        initial_root_pos = initial_body_poses[root_idx, :3]
        initial_root_quat = initial_body_poses[root_idx, 3:]
        q = self.state.q.at[:3].set(initial_root_pos)
        q = q.at[3:6].set(quat_wxyz_to_intrinsic_euler_xyz(initial_root_quat))
        self.state = self.state._replace(
            q=q,
            root_pos=initial_root_pos,
            root_quat_wxyz=initial_root_quat,
        )

        # TODO decide if it is better to separately JIT the JAX SCP step
        # versus calling the diff IK step in a loop here?
        # TODO move these constants somewhere else
        convergence_tol = 1e-3
        max_iters = 300
        # Adaptive stepsize for SCP -- help correct for potentially large
        # initial errors in the global positioning
        dts = np.logspace(-1, -2, max_iters)  # between 0.1 and 0.01
        print(f"[{type(self).__name__}]: Starting SCP initialization")
        for i in range(max_iters):
            self.state = self._jit_scp_step(
                self.functional_retargeter, initial_body_poses, self.state, dts[i]
            )
            converged = np.linalg.norm(self.state.root_vel) <= convergence_tol
            if converged:
                break
        print(
            f"[{type(self).__name__}]: SCP initialization "
            + ("converged " if converged else "did not converge ")
            + f"in {i} iterations"
        )
        # Reset some state info -- these are meaningless when thinking
        # about how these vary across SCP iterations
        self.state = self.state._replace(
            root_vel=jnp.zeros(3),
            root_omega=jnp.zeros(3),
            num_contact_frames=0,
        )
        self._is_initialized = True
        return self._outputs()

    def _outputs(self):
        # Helper function to output entries from internal state
        return (
            self.state.q,
            self.state.root_pos,
            self.state.root_quat_wxyz,
            self.state.root_vel,
            self.state.root_omega,
            self.state.contact_mode,
        )
