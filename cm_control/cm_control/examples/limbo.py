"""
Walking under an obstacle example

Unlike self-collision avoidance, this safety condition can NOT be modeled
during training time and thus has to be enforced at runtime.

This is also more complicated than the karate chop example as it involves
locomotion and repeated switching of the contact mode

STEPS
- Position the robot at an initial standing position with both feet planted on the ground
- Command a straight forward walking trajectory directly into the path of the obstacle
- Analyze how the CBF adjusts the reference motion (kinematics only) based on whether
  we filter based on the constrained kinematics or not
- Then add the policy on top and evaluate the performance.
    1. Kinematic only filter, with policy tracking
    2. Dynamic only filter, after the policy output
    3. Kinematic + dynamic

The dynamic filter component might not actually work that well here since the
contact modes are not necessarily perfectly known -- not quite sure

Should also experiment with representing the localization of the obstacle
in global frame versus in body root frame, as global localization is not
necessarily available
"""

import os

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)
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
from cm_control.utils.mujoco_utils import visualize_cylinder
from cm_control.utils.rotation_utils import Rx

# For simplicity assume that the cylinder is infinite in y
CYLINDER_X = 1.0
CYLINDER_Z = 1.4
# CYLINDER_RADIUS = 0.25 For original limbo example
CYLINDER_RADIUS = 0.45  # Better for crouching example
HEAD_INDEX = 23


# We have two recorded motions, one walks straight and one crouches while walking
# The straight walking motion leads to a "limbo" scenario when the constraint
# is enforced, whereas with the crouch walk it goes into a deeper crouch
LIMBO_OR_CROUCH = "limbo"


class LimboKinematicCBFConfig(BaseKinematicConfig):
    def __init__(self, robot):
        super().__init__(
            constrained=True,
            underactuated=False,
            robot=robot,
            use_naive_objective=False,
            solver_tol=1e-5,
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_1(self, z, *args, **kwargs):
        q = z
        collision_pos, collision_rad = self.robot.link_collision_data(q)
        # TODO consider enforcing this constraint on the whole-body geometry
        head_pos = collision_pos[HEAD_INDEX]
        head_rad = collision_rad[HEAD_INDEX]
        # Compute the distance to the cylinder in the XZ plane
        head_xz = jnp.array([head_pos[0], head_pos[2]])
        cylinder_xz = jnp.array([CYLINDER_X, CYLINDER_Z])
        dist_between_centers = jnp.linalg.norm(head_xz - cylinder_xz)
        return jnp.array([dist_between_centers - CYLINDER_RADIUS - head_rad])

    def alpha(self, h, *args, **kwargs):
        return 10.0 * h

    # NOTE: not implementing P/q since those get handled by the IK solver anyways
    # Could consider adjusting the IK objective to reduce effect on posture


class LimboDynamicCBFConfig(BaseDynamicConfig):
    def __init__(self, robot):
        super().__init__(
            constrained=True,
            underactuated=True,
            robot=robot,
            include_jdot=True,
            use_naive_objective=False,
            solver_tol=1e-5,
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_2(self, z, *args, **kwargs):
        q = z[: self.robot.num_joints]
        collision_pos, collision_rad = self.robot.link_collision_data(q)
        # TODO consider enforcing this constraint on the whole-body geometry
        head_pos = collision_pos[HEAD_INDEX]
        head_rad = collision_rad[HEAD_INDEX]
        # Compute the distance to the cylinder in the XZ plane
        head_xz = jnp.array([head_pos[0], head_pos[2]])
        cylinder_xz = jnp.array([CYLINDER_X, CYLINDER_Z])
        dist_between_centers = jnp.linalg.norm(head_xz - cylinder_xz)
        return jnp.array([dist_between_centers - CYLINDER_RADIUS - head_rad])

    def alpha(self, h, *args, **kwargs):
        return 10.0 * h

    def alpha_2(self, h, *args, **kwargs):
        return 10.0 * h


def get_contact_mode_mujoco(data, left_id, right_id):
    left_in_contact = False
    right_in_contact = False

    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 == left_id or contact.geom2 == left_id:
            left_in_contact = True
        if contact.geom1 == right_id or contact.geom2 == right_id:
            right_in_contact = True

    if left_in_contact and right_in_contact:
        return 3
    elif right_in_contact:
        return 2
    elif left_in_contact:
        return 1
    else:
        return 0


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
    folder = cm_root / "cm_ws" / "bags"
    if LIMBO_OR_CROUCH == "limbo":
        # ORIGINAL FORWARD WALK FILES
        # folder = "/home/dmorton/Downloads/wboscbf_rosbags/april 24-20260427T154111Z-3-001/april 24/cropped"
        # # file = "rosbag2_2026_04_24-15_05_01_walk_crop_1890_to_2285.pkl"
        file_2 = "rosbag2_2026_04_24-15_05_01_walk_crop_3756_to_4218.pkl"
        file_path = Path(folder) / file_2
    else:
        # NEW CROUCH WALK FILES
        # Note: twist2 seems to struggle a bit with these
        # folder = "/home/dmorton/Downloads/wboscbf_rosbags/may_6-20260506T205800Z-3-001/may_6/cropped"
        file = "rosbag2_2026_05_06-22_56_37_crouch_walk_crop_647_to_1336.pkl"
        file_path = Path(folder) / file

    robot = load_g1()
    if add_kinematic_cbf:
        kin_cbf_config = LimboKinematicCBFConfig(robot)
        kin_cbf = CBF.from_config(kin_cbf_config)
    else:
        kin_cbf = None
    if add_dynamic_cbf:
        dyn_cbf_config = LimboDynamicCBFConfig(robot)
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
    inner_retargeter = PicoToG1Retargeter(kin_cbf, robot)
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

    left_foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link"
    )
    right_foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link"
    )

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
            visualize_cylinder(
                viewer,
                pos=np.array([CYLINDER_X, 0, CYLINDER_Z]),
                rmat=Rx(np.pi / 2),
                radius=CYLINDER_RADIUS,
                size=0.25,  # Some arbitrary length. Short is good for visualization
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
                        contact_mode = get_contact_mode_mujoco(
                            data, left_foot_id, right_foot_id
                        )
                        pd_target = pd_filter.filter_from_mujoco_state(
                            pd_target, data.qpos, data.qvel, contact_mode
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
