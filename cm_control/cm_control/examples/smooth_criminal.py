"""
Michael Jackson's "Smooth Criminal" Example

We'll command an "anti gravity lean" as the kinematic reference
and use an online kinematic filter to maintain the COM of the
robot within the support polygon of the robot

This demo looks at three unique conditions:
- If we are NOT doing online retargeting and we already have a
    reference motion on the robot kinematics, how do we add an
    online kinematic filter on top of this? How does the objective
    structure differ between enforcing CBF constraints in retargeting/IK
    versus in a safety-filter type of scenario?
- How much does it matter that we use the constrained kinematics
    in the CBF construction (as opposed to the unconstrained kinematics?)
- Stability conditions in a fixed contact mode are not already covered
    by the other examples

Some notes:
- If the contact nullspace is not properly handled, the feet will
    lift off the floor!
- After we properly handle the constrained kinematics, the COM constraint
    will work but will violate joint limits. We can add joint limit CBFs
    to avoid this issue.
- With (1) proper handling of the constrained kinematics, (2) COM CBF,
    and (3) joint limits CBF, the motion is kinematically feasible but the
    standard min-norm joint-space objective leads to weird arm motion
"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import time
import argparse
from functools import partial
from typing import Tuple
from pathlib import Path

import jax
from jax import Array
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from frax import load_g1, Humanoid
from cbfpy import CBF

from cm_control.core.kinematic_configs import BaseKinematicConfig
from cm_control.assets import G1_XML
from cm_control.utils.transform_utils import (
    invert_transform,
    transform_to_pos_and_intrinsic_euler_xyz,
    transform_to_pos_and_quat_wxyz,
)
from cm_control.core.kinematic_objectives import P_qp, q_qp_from_P
from cm_control.core.support_polygon import support_polygon
from cm_control.utils.assorted_utils import mujoco_qpos_to_q, q_to_mujoco_qpos
from cm_control.utils.osc_utils import (
    nullspace_projection,
    selection_matrix,
    underactuated_contact_nullspace_projection,
)


def compute_antigravity_lean_motion(
    robot: Humanoid, duration: float, frequency: float
) -> Tuple[Array, Array]:
    """Helper function to compute the antigravity lean

    Will return both my representation of q for all timesteps, and mujoco's
    """
    # Get the robot in an initial configuration that looks like MJ's lean
    q_init = np.zeros(robot.num_joints)
    left_elbow_idx = robot.joint_name_to_index["left_elbow_joint"]
    right_elbow_idx = robot.joint_name_to_index["right_elbow_joint"]
    q_init[left_elbow_idx] = 1.4
    q_init[right_elbow_idx] = 1.4

    left_shoulder_roll_idx = robot.joint_name_to_index["left_shoulder_roll_joint"]
    right_shoulder_roll_idx = robot.joint_name_to_index["right_shoulder_roll_joint"]
    q_init[left_shoulder_roll_idx] = 0.2
    q_init[right_shoulder_roll_idx] = -0.2

    left_ankle_idx = robot.joint_name_to_index["left_ankle_pitch_joint"]
    right_ankle_idx = robot.joint_name_to_index["right_ankle_pitch_joint"]

    max_pitch = 0.7 * robot.joint_lower_limits[left_ankle_idx]
    num_timesteps = duration * frequency
    pitches = np.linspace(0.0, max_pitch, num_timesteps)

    def _compute_step(_pitch):
        q = (
            jnp.asarray(q_init)
            .at[np.array([left_ankle_idx, right_ankle_idx])]
            .set(_pitch)
        )
        # Determine the transform to apply to the root to keep the feet fixed
        # Assume that the left foot is the "new root" of the kinematic tree
        left_foot_tf = robot.left_foot_transform(q)
        pelvis_tf = robot.joint_to_world_transforms(q)[5]
        pelvis_to_foot_tf = invert_transform(left_foot_tf) @ pelvis_tf
        new_q = jnp.concatenate(
            [
                transform_to_pos_and_intrinsic_euler_xyz(pelvis_to_foot_tf),
                q[-29:],
            ]
        )
        new_mujoco_qpos = jnp.concatenate(
            [transform_to_pos_and_quat_wxyz(pelvis_to_foot_tf), q[-29:]]
        )
        return new_q, new_mujoco_qpos

    @jax.jit
    def _jit_compute_motion(_pitches):
        return jax.vmap(_compute_step, in_axes=(0,))(_pitches)

    my_qs, mujoco_qposes = _jit_compute_motion(pitches)
    return my_qs, mujoco_qposes


@jax.tree_util.register_static
class StabilityKinematicCBFConfig(BaseKinematicConfig):
    def __init__(self, robot, q_init):
        # NOTE: We'll treat the support polygon as static for this example,
        # based on the initial configuration of the robot
        self.A_support, self.b_support = support_polygon(robot, q_init, contact_mode=3)

        super().__init__(
            constrained=True,
            underactuated=True,  # !!
            robot=robot,
            use_naive_objective=False,  # Doesn't matter what this flag is because we overwrite P/q
            solver_tol=1e-5,
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_1(self, z, *args, **kwargs):
        # The support polygon is defined as Ax<=b with Ax-b <= 0 for x in the polygon
        # If we negate this, b - Ax gives us the distance to the boundary
        # Any padded redundant constraints evaluate to 1 - 0x = 1.0 (safe)
        q = z
        com_pos_xy = self.robot.center_of_mass(q)[:2]
        h_com = self.b_support - self.A_support @ com_pos_xy
        # Add joint limits constraint to keep the knees from exceeding their limit
        h_joint_lower_limits = q - jnp.asarray(self.robot.joint_lower_limits)
        h_joint_upper_limits = jnp.asarray(self.robot.joint_upper_limits) - q
        return jnp.concatenate([h_com, h_joint_lower_limits, h_joint_upper_limits])

    def alpha(self, h, *args, **kwargs):
        return 10.0 * h

    # What is the task-consistent objective in this case?
    # We care about the body remaining straight if possible
    # The task-consistent objective actually is a min-norm objective BUT
    # defined on the UNDERACTUATED inputs!
    # i.e. minimize ||qdot_actuated - qdot_actuated_nominal||2_2
    def P(self, z, u_des, *args, **kwargs):
        return jnp.eye(self.m)

    def q(self, z, u_des, *args, **kwargs):
        P = self.P(z, u_des, *args, **kwargs)
        return -P @ u_des


def main():
    # Set up the sim
    xml_path = str(G1_XML)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    robot = load_g1()
    real_time = True

    # Compute the motion
    duration = 5
    frequency = 50
    dt = 1 / frequency
    qs, qposes = compute_antigravity_lean_motion(robot, duration, frequency)
    num_steps = qs.shape[0]

    # Update mujoco to match initial state
    data.qpos[:] = qposes[0]
    mujoco.mj_forward(model, data)

    # Construct the CBF
    cbf_config = StabilityKinematicCBFConfig(robot, qs[0])
    cbf = CBF.from_config(cbf_config)

    @jax.jit
    def filter_and_integrate(q, q_desired):
        # Robot terms
        joint_transforms = robot.joint_to_world_transforms(q)
        J_c_L = robot._left_foot_jacobian(joint_transforms)
        J_c_R = robot._right_foot_jacobian(joint_transforms)
        M = robot._mass_matrix(joint_transforms)
        M_inv = robot.mass_matrix_inverse(M)
        J_c = jnp.vstack([J_c_L, J_c_R])
        N_c = nullspace_projection(J_c, M_inv)
        S = selection_matrix(robot.num_actuated_joints)
        SNc_bar = underactuated_contact_nullspace_projection(J_c, M_inv, S)
        # Simple unsafe joint-space PD control to track the reference
        kp = 1.0
        qdot_unsafe = kp * (q_desired - q)
        qdot_act_safe = cbf.safety_filter(q, qdot_unsafe[6:], contact_mode=3)
        qdot_safe = SNc_bar @ qdot_act_safe
        # Integrate while properly handling the constrained kinematics!
        qdot = N_c @ qdot_safe
        return q + qdot * dt

    # Initial call for jit compilation
    print("Jitting...")
    jax.block_until_ready(filter_and_integrate(qs[0], qs[0]))
    print("JIT complete")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_q = qs[0]
            for i in range(num_steps):
                start_time = time.time()
                current_q = filter_and_integrate(current_q, qs[i])
                data.qpos[:] = q_to_mujoco_qpos(current_q)
                mujoco.mj_forward(model, data)
                if not viewer.is_running():
                    break
                viewer.sync()
                elapsed = time.time() - start_time
                if real_time and elapsed < dt:
                    time.sleep(dt - elapsed)
                # print(cbf.h(current_q, contact_mode=3))
            else:
                print("QPOS at end of motion:")
                print(
                    np.array2string(
                        data.qpos,
                        separator=",",
                        max_line_width=np.inf,
                        formatter={"all": lambda x: str(x)},
                    )
                )


if __name__ == "__main__":
    main()
