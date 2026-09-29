"""Kinematic retargeting"""

from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from cbfpy import CBF
import elastiqp.jax

from frax import Humanoid
from cm_control.utils.rotation_utils import orientation_error_3D
from cm_control.utils.osc_utils import (
    contact_mask_from_mode,
    masked_nullspace_projection,
)


@jax.tree_util.register_static
class KinematicRetargeter:
    """Kinematic retargeter

    Args:
        cbf (Optional[CBF]): CBF configuration for constrained kinematic retargeting.
            Set to None if no CBF is desired.
        robot (Humanoid): Humanoid to retarget (i.e. Unitree g1). Currently,
            we assume that we are retargeting a floating-base humanoid
        idxs (Array): Indices of the links of interest for the retargeting tasks.
            Shape = (num_links_of_interest,)
        kp_pos (Union[float, int, Array]): Proportional gain on positional errors
            between the current and target link positions. Scalar or an array
            of shape (num_links_of_interest,)
        kp_rot (Union[float, int, Array]): Proportional gain on rotational errors
            between the current and target link rotations. Scalar or an array
            of shape (num_links_of_interest,)
        weight_pos (Union[float, int, Array]): Objective weighting for positional
            retargeting tasks. Scalar or an array of shape (num_links_of_interest,)
        weight_rot (Union[float, int, Array]): Objective weighting for rotational
            retargeting tasks. Scalar or an array of shape (num_links_of_interest,)
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
        idxs: Array,
        kp_pos: Union[float, int, Array],
        kp_rot: Union[float, int, Array],
        weight_pos: Union[float, int, Array],
        weight_rot: Union[float, int, Array],
        stepsize: float = 1.0,
        convergence_tol: float = 1e-3,
        max_iters: int = 50,
        regularization: Optional[float] = 1e-12,
        solver_tol: float = 1e-5,
    ):
        self.idxs = jnp.asarray(idxs)
        self.num_links_of_interest = self.idxs.shape[0]

        assert isinstance(cbf, CBF) or cbf is None
        self.cbf = cbf
        assert isinstance(robot, Humanoid) and robot.includes_floating_dof
        self.robot = robot
        # TODO: Should clarify the relationship between kp, stepsize, and dt
        # Stepsize: used in SCP
        # dt: Used for integration of qdot from diff IK
        # kp: Gains, used for diff IK
        # All three intermingle with eachother. If kp = 1, for instance, a SCP
        # stepsize of 1 works well. But if kp = 10, stepsize of 1 is too much.
        # Perhaps one of these parameters should be a function of the other two
        assert isinstance(stepsize, float) and stepsize > 0.0
        if stepsize > 1.0:
            print(
                f"WARNING: Desired stepsize ({stepsize}) is greater than 1.0  -- this is not recommended"
            )
        self.stepsize = stepsize
        assert isinstance(convergence_tol, float) and convergence_tol > 0.0
        self.convergence_tol = convergence_tol
        assert isinstance(max_iters, int) and max_iters > 0
        self.max_iters = max_iters
        if isinstance(kp_pos, (float, int)):
            kp_pos = float(kp_pos) * jnp.ones(self.num_links_of_interest)
        else:
            assert kp_pos.shape == (self.num_links_of_interest,)
        self.kp_pos = kp_pos

        if isinstance(kp_rot, (float, int)):
            kp_rot = float(kp_rot) * jnp.ones(self.num_links_of_interest)
        else:
            assert kp_rot.shape == (self.num_links_of_interest,)
        self.kp_rot = kp_rot

        if isinstance(weight_pos, (float, int)):
            weight_pos = float(weight_pos) * jnp.ones(self.num_links_of_interest)
        else:
            assert weight_pos.shape == (self.num_links_of_interest,)
        self.weight_pos = weight_pos

        if isinstance(weight_rot, (float, int)):
            weight_rot = float(weight_rot) * jnp.ones(self.num_links_of_interest)
        else:
            assert weight_rot.shape == (self.num_links_of_interest,)
        self.weight_rot = weight_rot

        if isinstance(regularization, float):
            assert 0 <= regularization <= 1.0
        else:
            assert regularization is None
        self.regularization = regularization

        assert isinstance(solver_tol, float) and solver_tol > 0
        self.solver_tol = solver_tol

        self.v_min = -jnp.asarray(self.robot.joint_max_velocities)
        self.v_max = jnp.asarray(self.robot.joint_max_velocities)
        self.m = self.robot.num_joints  # Assume fully actuated

        twist_weights_per_link = []
        for i in range(self.num_links_of_interest):
            twist_weights_per_link.append(
                jnp.concatenate(
                    [weight_pos[i] * jnp.ones(3), weight_rot[i] * jnp.ones(3)]
                )
            )
        self.ws = jnp.concatenate(twist_weights_per_link)[:, None]

    @jax.jit
    def diff_ik_step(
        self,
        q: Array,
        desired_link_positions: Array,
        desired_link_rotations: Array,
        *cbf_args,
        **cbf_kwargs,
    ) -> Array:
        # TODO figure out if retargeting should be done based on the link inertial frame or joint frame
        # Assuming link inertial frame for now... otherwise use joint jacobians method
        joint_transforms = self.robot.joint_to_world_transforms(q)
        all_link_Jvs = self.robot._link_linear_jacobians(joint_transforms)
        all_link_Jws = self.robot._link_angular_jacobians(joint_transforms)
        all_link_tfs = self.robot._link_to_world_transforms(joint_transforms)
        all_link_positions = all_link_tfs[:, :3, 3]
        all_link_rotations = all_link_tfs[:, :3, :3]
        selected_link_positions = all_link_positions[self.idxs]
        selected_link_rotations = all_link_rotations[self.idxs]

        selected_link_Jvs = all_link_Jvs[self.idxs]
        selected_link_Jws = all_link_Jws[self.idxs]
        # Stack linear and angular jacobians to get 6xnq jacobians for each link
        # Shape: (num_links, 6, num_joints)
        link_Js = jnp.stack([selected_link_Jvs, selected_link_Jws], axis=1)
        # Now stack along the first axis to handle all tasks
        # Shape: (6 * num_links, num_joints)
        stacked_link_Js = jnp.reshape(link_Js, (-1, self.robot.num_joints))

        # Compute the desired twist to reduce the error dynamics for every link
        def compute_twist(cur_pos, des_pos, cur_rot, des_rot, kp_pos_val, kp_rot_val):
            vel_err_dyn = -kp_pos_val * (cur_pos - des_pos)
            omega_err_dyn = -kp_rot_val * orientation_error_3D(cur_rot, des_rot)
            return jnp.concatenate([vel_err_dyn, omega_err_dyn])

        stacked_twists = jax.vmap(compute_twist)(
            selected_link_positions,
            desired_link_positions,
            selected_link_rotations,
            desired_link_rotations,
            self.kp_pos,
            self.kp_rot,
        ).reshape(-1)

        assert stacked_twists.shape[0] == stacked_link_Js.shape[0]

        # TODO: Check to see if the regularization term is necessary
        # and include as an input to the class
        WJ = self.ws * stacked_link_Js
        P_qp = WJ.T @ WJ
        if self.regularization is not None:
            P_qp = P_qp + self.regularization * jnp.eye(self.robot.num_joints)
        q_qp = -stacked_twists.T @ (self.ws * WJ)

        h_input_cons = jnp.concatenate(
            [jnp.asarray(self.v_max), -jnp.asarray(self.v_min)]
        )
        G_input_cons = jnp.vstack([jnp.eye(self.m), -jnp.eye(self.m)])

        if self.cbf is not None:
            z = q  # CBF dynamics state is just joint positions
            hz, lfh = self.cbf.h_and_Lfh(z, *cbf_args, **cbf_kwargs)
            h_cbf_qp = self.cbf.alpha(hz, *cbf_args, **cbf_kwargs) + lfh
            G_cbf_qp = -self.cbf.Lgh(z, *cbf_args, **cbf_kwargs)

            G_qp = jnp.vstack([G_cbf_qp, G_input_cons])
            h_qp = jnp.concatenate([h_cbf_qp, h_input_cons])

            x_qp = elastiqp.jax.solve(
                P_qp,
                q_qp,
                G_qp,
                h_qp,
                penalty=jnp.asarray(self.cbf.constraint_relaxation_penalties),
                eps_abs=self.solver_tol,
            ).x
        else:
            G_qp = G_input_cons
            h_qp = h_input_cons
            x_qp = elastiqp.jax.solve(
                P_qp,
                q_qp,
                G_qp,
                h_qp,
                penalty=1e3,  # TODO make input
                eps_abs=self.solver_tol,
            ).x
        return x_qp  # this is qdot

    @jax.jit
    def scp_retarget(
        self,
        q_init: Array,
        desired_link_positions: Array,
        desired_link_rotations: Array,
        *cbf_args,
        **cbf_kwargs,
    ) -> Array:
        """Run kinematic retargeting

        Args:
            q_init (Array): Current joint positions of the robot, shape (num_joints,)
            desired_link_positions (Array): Desired world-frame positions of the links
                of the retargeting robot/human, shape (num_links_of_interest, 3)
            desired_link_rotations (Array): Desired world-frame rotation matrices of the links
                of the retargeting robot/human, shape (num_links_of_interest, 3, 3)

        Returns:
            Array: Retargeted joint angles, shape (num_joints,)
        """

        ### Handle inputs ###

        q_init = jnp.asarray(q_init)
        assert q_init.shape == (self.robot.num_joints,)

        desired_link_positions = jnp.asarray(desired_link_positions)
        assert desired_link_positions.shape == (self.num_links_of_interest, 3)

        desired_link_rotations = jnp.asarray(desired_link_rotations)
        assert desired_link_rotations.shape == (self.num_links_of_interest, 3, 3)

        ### Retarget ###

        iters_init = 0
        converged_init = 0

        def scp_inner(inputs: Tuple[Array, int, int]) -> Tuple[Array, int, int]:
            # Unpack inputs
            q, converged, iters = inputs
            # Run diff IK
            q_dot = self.diff_ik_step(
                q,
                desired_link_positions,
                desired_link_rotations,
                *cbf_args,
                **cbf_kwargs,
            )
            # Simple integration of qdot and convergence check
            # TODO decide on how I'm handling the stepsize here!
            q_next = self.integrate_qdot(q, q_dot, self.stepsize)
            converged = jnp.where(
                jnp.all(jnp.abs(q - q_next) < self.convergence_tol), 1, 0
            )
            return q_next, converged, iters + 1

        def continuation_criteria(inputs: Tuple[Array, int, int]) -> bool:
            """Inner retargeting loop continuation check. True if we should continue to iterate"""
            q, converged, iters = inputs  # Unpack
            return jnp.logical_and(converged == 0, iters < self.max_iters)

        # TODO decide if we want to use the other outputs
        q_final, converged_final, iters_final = jax.lax.while_loop(
            continuation_criteria, scp_inner, (q_init, converged_init, iters_init)
        )
        return q_final, converged_final, iters_final

    def integrate_qdot(
        self, q: Array, qdot: Array, dt: float, contact_mode: Optional[int] = None
    ) -> Array:
        """Integrate a joint velocity (from diff IK)

        Currently this is extremely simple Euler integration, but
        we could use a better integrator in the future

        Args:
            q (Array): Joint angles, shape (num_joints,)
            qdot (Array): Joint velocities, shape (num_joints,)
            dt (float): Timestep
            contact_mode (int, optional): Contact mode. Defaults to None
                (integrate without considering the constrained kinematics)

        Returns:
            Array: Next joint angles, shape (num_joints,)
        """
        if contact_mode is None:
            return q + qdot * dt

        joint_transforms = self.robot.joint_to_world_transforms(q)
        J_lf = self.robot._left_foot_jacobian(joint_transforms)
        J_rf = self.robot._right_foot_jacobian(joint_transforms)
        M = self.robot._mass_matrix(joint_transforms)
        M_inv = self.robot.mass_matrix_inverse(M)

        # With no contact, N_c = identity (simple unconstrained integration)
        mask = contact_mask_from_mode(contact_mode)
        N_c = masked_nullspace_projection(jnp.vstack([J_lf, J_rf]), mask, M_inv)
        return q + N_c @ qdot * dt
