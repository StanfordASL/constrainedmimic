import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import time
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from frax import load_g1

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


def run_demo(kinematics_only: bool):
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
    file = "rosbag2_2026_04_24-15_05_01_walk_crop_1876_to_2860.pkl"
    file_path = Path(folder) / file

    robot = load_g1()
    kin_cbf = None
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
            for i in range(num_motion_steps):
                mimic_q = retargeting_data["q"][i]
                mimic_pos = retargeting_data["pos"][i]
                mimic_quat = retargeting_data["quat"][i]
                mimic_vel = retargeting_data["vel"][i]
                mimic_omega = retargeting_data["omega"][i]
                # mimic_contact_mode = retargeting_data["contact_mode"][i]

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
    args = parser.parse_args()
    kinematics_only = args.kinematics_only
    print("=" * 20)
    print(f"[{Path(__file__).name}]: Starting demo.")
    print("Configuration options: ")
    print(f"{kinematics_only=}")
    print("=" * 20)
    run_demo(kinematics_only)


if __name__ == "__main__":
    main()
