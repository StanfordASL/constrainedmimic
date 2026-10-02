"""
Self-collision avoidance example

This example shows "For a safety condition that can be easily modeled at training
time and included in the rewards, how much does adding a safety filter add or
detract from the performance?

STEPS
- Position the robot at an initial standing position with both feet planted on the ground
- Command a reference motion that causes the arm to hit the leg (or some other body)
- Analyze the effects of adding different types of CBFs on purely the kinematic level
- Then add the policy on top and evaluate the following:
    1. Kinematic only filter, with policy tracking
    2. Dynamic only filter, after the policy output
    3. Kinematic + dynamic
- Measure how often a collision occurs, how close we get to a collision, and impact force?

It might also be worthwhile to compare the kinematic motion to what is output from Mink
or Pink and see if a non-CBF approach (Mink) gives a worse output (for instance, maybe
this gives a more jerky stopping behavior near the boundary of safety which is harder
for the policy to track)
"""

# TODO
# The dynamic CBF doesn't work great in this example. Possibly, it's because there
# are so many self collision pairs currently. Maybe I can simplify the SC model?
# Alternatively I could try representing things with capsules but then I need to
# actually define the distance metric for that

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
from cm_control.core.dynamic_objectives import P_qp, q_qp_from_P
from cm_control.core.dynamic_filter_utils import PDFilter


class SelfCollisionKinematicCBFConfig(BaseKinematicConfig):
    def __init__(self, robot):
        super().__init__(
            constrained=True,
            underactuated=False,
            robot=robot,
            use_naive_objective=False,
            solver_tol=1e-5,  # Same as default
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_1(self, z, *args, **kwargs):
        q = z
        eps = 1e-3
        return self.robot.self_collision_distances(q) - eps

    def alpha(self, h, *args, **kwargs):
        return 5.0 * h

    # NOTE: not implementing P/q since those get handled by the IK solver anyways
    # Could consider adjusting the IK objective to reduce effect on posture


class SelfCollisionDynamicCBFConfig(BaseDynamicConfig):
    def __init__(self, robot):
        super().__init__(
            constrained=True,
            underactuated=True,
            robot=robot,
            include_jdot=True,
            use_naive_objective=False,
            solver_tol=1e-8,  # Same as default
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_2(self, z, *args, **kwargs):
        q = z[: self.robot.num_joints]
        eps = 1e-3
        return self.robot.self_collision_distances(q) - eps

    def alpha(self, h, *args, **kwargs):
        return 15.0 * h

    def alpha_2(self, h, *args, **kwargs):
        return 15.0 * h

    # NOTE: P/q are inherited


def run_demo(
    kinematics_only: bool,
    add_kinematic_cbf: bool,
    add_dynamic_cbf: bool,
    save_qposes: bool,
):
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
    # folder = "/home/dmorton/Downloads/wboscbf_rosbags/april 24-20260427T154111Z-3-001/april 24/cropped"
    file_1 = "rosbag2_2026_04_24-15_07_06_self_collision_crop_1305_to_2380.pkl"
    # file_2 = "rosbag2_2026_04_24-15_07_06_self_collision_crop_2484_to_3236.pkl"
    file_path = Path(folder) / file_1

    robot = load_g1()
    if add_kinematic_cbf:
        kin_cbf_config = SelfCollisionKinematicCBFConfig(robot)
        kin_cbf = CBF.from_config(kin_cbf_config)
    else:
        kin_cbf = None
    if add_dynamic_cbf:
        dyn_cbf_config = SelfCollisionDynamicCBFConfig(robot)
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

    # Reset the state of the sim to the first state in the retargeting data
    data.qpos[:3] = retargeting_data["pos"][0]
    data.qpos[3:7] = retargeting_data["quat"][0]
    data.qpos[7:] = retargeting_data["q"][0, 6:]
    mujoco.mj_forward(model, data)

    qposes_data = []
    saved = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
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
                    if save_qposes and not saved:
                        qposes_data.append(np.copy(data.qpos))
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
                    # Update sim (1000 Hz)
                    for i in range(decimation):
                        if add_dynamic_cbf and i % 4 == 0:
                            # HACK (somewhat) assume that we can run the dynamic cbf at 250hz ish
                            # This should actually be reasonable given the numbers I've been seeing
                            # TODO USE THE ACTUAL CONTACT MODE FROM THE SIM STATE
                            contact_mode = 3
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
                        # Compute PD control for joint torques (clipped to actuator limits)
                        torque = np.clip(
                            twist2_config.sim_kps * (pd_target - data.qpos[7:])
                            + twist2_config.sim_kds * -data.qvel[6:],
                            -g1_config.joint_max_torques,
                            g1_config.joint_max_torques,
                        )
                        data.ctrl[:] = torque
                        mujoco.mj_step(model, data)
                        if save_qposes and not saved:
                            qposes_data.append(np.copy(data.qpos))

                # Handle mujoco viewer updates
                if not viewer.is_running():
                    break
                viewer.sync()
                elapsed = time.time() - start_time
                if real_time and elapsed < policy_dt:
                    time.sleep(policy_dt - elapsed)
            else:
                print(f"[{Path(__file__).name}]: Motion complete.")
                if save_qposes and not saved:
                    print("Saving qposes data...")
                    filename = (
                        "self_collision_qposes_"
                        + ("kinematics_only_" if kinematics_only else "")
                        + ("add_kin_cbf_" if add_kinematic_cbf else "")
                        + ("add_dyn_cbf_" if add_dynamic_cbf else "")
                    )
                    np.save(filename, np.asarray(qposes_data))
                    saved = True
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
    parser.add_argument("--save-qposes", default=False, action="store_true")
    args = parser.parse_args()
    kinematics_only = args.kinematics_only
    add_kinematic_cbf = args.add_kinematic_cbf
    add_dynamic_cbf = args.add_dynamic_cbf
    save_qposes = args.save_qposes
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
    print(f"{save_qposes=}")
    print("=" * 20)
    run_demo(kinematics_only, add_kinematic_cbf, add_dynamic_cbf, save_qposes)


if __name__ == "__main__":
    main()
