"""
Karate chop example

This is an example of a safety condition that can NOT be modeled
during training time and must be enforced at runtime

This is also a good way to tune the objective and kinematics/dynamics
used for the CBF, as it will be very easy to tell if enforcing the
constraint induces excess motion in the null space of the constrained
direction. And, if the contact constraints on the feet are not properly
handled, it could even lead to them coming off the ground

STEPS
- Position the robot at an initial standing position with both feet planted on the ground
- Point the right arm outwards and command a downwards motion directly into a planar limit
  on the hand position
- Analyze how the CBF adjusts the reference motion (kinematics only) based on whether
  we filter based on the constrained kinematics or not
- Then add the policy on top and evaluate the performance.
    1. Kinematic only filter, with policy tracking
    2. Dynamic only filter, after the policy output
    3. Kinematic + dynamic

On hardware, we could implement this with a nice demo with foam board, where
we command the hand to chop into the foam board and success should occur
if the foam board is unbroken. What we're likely to see is:
- Baseline policy with no safety will obviously break through
- Adding the constraint only on the kinematics breaks the board because the
  policy is unable to perfectly track the kinematic reference
- Adding the constraint only on the dynamics might work? But also might fail
  if there is a bit of model mismatch. I suspect there might be some tradeoffs
  here because while we likely know the upper body dynamics pretty well,
  the lower body dynamics (due to the closed chain and approximated contact
  models) are less known
- Adding the bilevel constraint will likely be the best balance

Like the self-collision demo, we could also use something link mink as another
baseline comparison with a more aggressive hard constraint than the CBF
"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import time
import argparse
from pathlib import Path

import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from frax import load_g1
from cbfpy import CBF

from cm_control.assets import G1_XML
from cm_control.twist2_utils import twist2_config
from cm_control.config import g1_config
from cm_control.retargeting.motion_processing import load_and_retarget_motion
from cm_control.retargeting.pico_to_g1_retarget import (
    PicoToG1Retargeter,
    StatefulRetargeter,
)
from cm_control.twist2_utils.motion_tracking import (
    FunctionalMotionTracker,
    StatefulMotionTracker,
)
from cm_control.core.kinematic_configs import BaseKinematicConfig
from cm_control.core.dynamic_configs import BaseDynamicConfig
from cm_control.core.actuators import RobotActuatorModel
from cm_control.core.dynamic_filter_utils import PDFilter
from cm_control.utils.mujoco_utils import visualize_box


Z_MIN = 0.75
HAND_RADIUS = 0.06237264  # Directly from collision model


class KarateChopKinematicCBFConfig(BaseKinematicConfig):
    def __init__(self, robot):
        super().__init__(
            constrained=True,
            underactuated=False,
            robot=robot,
            use_naive_objective=False,
            solver_tol=1e-7,
            init_args=None,  # Same as default
            init_kwargs={"contact_mode": 3},
        )

    def h_1(self, z, *args, **kwargs):
        q = z
        collision_pos, collision_rad = self.robot.link_collision_data(q)
        right_hand_index = 44
        right_hand_pos = collision_pos[right_hand_index]
        right_hand_rad = collision_rad[right_hand_index]
        right_hand_z = right_hand_pos[2]
        return jnp.array([right_hand_z - Z_MIN - right_hand_rad])

    def alpha(self, h, *args, **kwargs):
        return 4.0 * h

    # NOTE: not implementing P/q since those get handled by the IK solver anyways
    # Could consider adjusting the IK objective to reduce effect on posture


class KarateChopDynamicCBFConfig(BaseDynamicConfig):
    def __init__(self, robot):
        super().__init__(
            constrained=True,
            underactuated=True,
            robot=robot,
            include_jdot=True,
            use_naive_objective=False,
            solver_tol=1e-5,  # Same as default
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_2(self, z, *args, **kwargs):
        q = z[: self.robot.num_joints]
        collision_pos, collision_rad = self.robot.link_collision_data(q)
        right_hand_index = 44
        right_hand_pos = collision_pos[right_hand_index]
        right_hand_rad = collision_rad[right_hand_index]
        right_hand_z = right_hand_pos[2]
        return jnp.array([right_hand_z - Z_MIN - right_hand_rad])

    def alpha(self, h, *args, **kwargs):
        return 10.0 * h

    def alpha_2(self, h, *args, **kwargs):
        return 10.0 * h

    # NOTE: P/q are inherited


def run_demo(kinematics_only: bool, add_kinematic_cbf: bool, add_dynamic_cbf: bool):
    # Configuration values
    physics_freq = 1000
    policy_freq = 50
    decimation = int(physics_freq / policy_freq)
    policy_dt = 1 / policy_freq
    replay_freq = policy_freq
    real_time = True
    restart_motion = True

    # Filepaths to XR data
    cm_root = Path(__file__).parents[3]
    folder = cm_root / "cm_control" / "cm_control" / "assets" / "data"
    file = "rosbag2_2026_05_06-13_49_25_crop_2046_to_2450.pkl"
    file_path = folder / file

    robot = load_g1()
    if add_kinematic_cbf:
        kin_cbf_config = KarateChopKinematicCBFConfig(robot)
        kin_cbf = CBF.from_config(kin_cbf_config)
    else:
        kin_cbf = None
    if add_dynamic_cbf:
        dyn_cbf_config = KarateChopDynamicCBFConfig(robot)
        dyn_cbf = CBF.from_config(dyn_cbf_config)
        actuator_model = RobotActuatorModel(
            twist2_config.sim_kps, twist2_config.sim_kds, g1_config.joint_max_torques
        )
        pd_filter = PDFilter(dyn_cbf, actuator_model)
    else:
        dyn_cbf = None
        actuator_model = None
        pd_filter = None

    # TODO rework how these are constructed
    inner_retargeter = PicoToG1Retargeter(
        kin_cbf, robot, use_constrained_integration=True, force_double_support=True
    )
    retargeter = StatefulRetargeter(inner_retargeter)

    # Motion tracker
    functional_tracker = FunctionalMotionTracker()
    motion_tracker = StatefulMotionTracker(functional_tracker)

    # Load and retarget the motion to the robot kinematics
    retargeting_data = load_and_retarget_motion(file_path, replay_freq, retargeter)
    num_motion_steps = retargeting_data["q"].shape[0]

    # Set up the sim
    xml_path = str(G1_XML)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Set up the initial state of the sim to match the first state in the
    # retargeting data. Also, shift the position/orientation back to the origin
    # so that we can line it up nicely with the obstacle
    data.qpos[:2] = np.zeros(2)
    data.qpos[2] = retargeting_data["pos"][0, 2]
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:] = retargeting_data["q"][0, 6:]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Create a visual box element to show the height constraint
            visualize_box(
                viewer,
                pos=np.array([0.75, 0, (Z_MIN - HAND_RADIUS) / 2]),
                rmat=np.eye(3),
                size=(0.5, 0.5, (Z_MIN - HAND_RADIUS) / 2),
                rgba=(1, 0, 0, 0.5),
                reset_ngeom=True,
            )
            for i in range(num_motion_steps):
                mimic_q = retargeting_data["q"][i]
                mimic_pos = retargeting_data["pos"][i]
                mimic_quat = retargeting_data["quat"][i]
                mimic_vel = retargeting_data["vel"][i]
                mimic_omega = retargeting_data["omega"][i]
                mimic_contact_mode = retargeting_data["contact_mode"][i]

                start_time = time.time()
                if kinematics_only:
                    data.qpos[7:] = mimic_q[6:]
                    data.qpos[:3] = mimic_pos
                    data.qpos[3:7] = mimic_quat
                    mujoco.mj_forward(model, data)
                else:  # Dynamics
                    # Compute PD target (50 Hz)
                    pd_target = motion_tracker.step(
                        robot_quat_wxyz=data.qpos[3:7],
                        robot_omega_body=data.qvel[3:6],
                        robot_q=data.qpos[-29:],
                        robot_qd=data.qvel[-29:],
                        mimic_pos=mimic_pos,
                        mimic_quat=mimic_quat,
                        mimic_vel=mimic_vel,
                        mimic_omega=mimic_omega,
                        mimic_q=mimic_q[-29:],
                    )
                    # TODO: Decide if this should be called at the same frequency
                    # as the policy or if it should be faster (say 100 hz)
                    if add_dynamic_cbf:
                        # TODO USE THE ACTUAL CONTACT MODE FROM THE SIM STATE
                        pd_target = pd_filter.filter_from_mujoco_state(
                            pd_target, data.qpos, data.qvel, mimic_contact_mode
                        )
                        # HACK -- update the last action state after filtering
                        motion_tracker._last_action = (
                            1 / motion_tracker.functional_tracker.action_scale
                        ) * (
                            pd_target
                            - motion_tracker.functional_tracker.default_dof_pos
                        )
                    # Update sim (1000 Hz)
                    for _ in range(decimation):
                        # Compute PD control for joint torques (clipped to actuator limits)
                        torque = np.clip(
                            twist2_config.sim_kps * (pd_target - data.qpos[7:])
                            + twist2_config.sim_kds * -data.qvel[6:],
                            -g1_config.joint_max_torques,
                            g1_config.joint_max_torques,
                        )
                        data.ctrl[:] = torque
                        mujoco.mj_step(model, data)

                # Handle mujoco viewer updates
                if not viewer.is_running():
                    break
                viewer.sync()
                elapsed = time.time() - start_time
                if real_time and elapsed < policy_dt:
                    time.sleep(policy_dt - elapsed)
            else:
                print(f"[{Path(__file__).name}]: Motion complete.")
            if restart_motion and viewer.is_running():
                print(f"[{Path(__file__).name}]: Replaying motion...")
            else:
                break


def main():
    # Parse + check args before launching the demo
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinematics-only", default=False, action="store_true")
    parser.add_argument("--add-kinematic-cbf", default=False, action="store_true")
    parser.add_argument("--add-dynamic-cbf", default=False, action="store_true")
    args = parser.parse_args()
    kinematics_only = args.kinematics_only
    add_kinematic_cbf = args.add_kinematic_cbf
    add_dynamic_cbf = args.add_dynamic_cbf
    if add_dynamic_cbf and kinematics_only:
        raise ValueError(
            "The 'Add Dynamic CBF' argument was specified at the same time as 'Kinematics Only'.\n"
        )
    print("=" * 20)
    print(f"[{Path(__file__).name}]: Starting demo.")
    print("Configuration options: ")
    print(f"{kinematics_only=}")
    print(f"{add_kinematic_cbf=}")
    print(f"{add_dynamic_cbf=}")
    print("=" * 20)
    run_demo(kinematics_only, add_kinematic_cbf, add_dynamic_cbf)


if __name__ == "__main__":
    main()
