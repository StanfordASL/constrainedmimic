"""
Standing example

This script just commands the motion tracking policy to hold a
a static standing pose. This is primarily used to debug and
interact with the motion tracking policy
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from cm_control.assets import G1_XML
from cm_control.twist2_utils import twist2_config
from cm_control.config import g1_config
from cm_control.twist2_utils.motion_tracking import (
    FunctionalMotionTracker,
    StatefulMotionTracker,
)

# fmt: off
INITIAL_QPOS = np.concatenate([
    # This choice of the z height puts the feet perfectly in contact on init
    np.array([0, 0, 0.779]),
    np.array([1, 0, 0, 0]),
    np.array([
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # Left leg
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # Right leg
        0.0, 0.0, 0.0,  # Torso
        0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0,  # Left arm
        0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0,  # Right arm
    ])
])
# fmt: on


def initialize_environment():
    xml_path = str(G1_XML)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[:] = INITIAL_QPOS
    mujoco.mj_forward(model, data)
    return model, data


def apply_pd_control(data: mujoco.MjData, pd_target: np.ndarray):
    torque = (
        twist2_config.sim_kps * (pd_target - data.qpos[7:])
        + twist2_config.sim_kds * -data.qvel[6:]
    )
    torque = np.clip(torque, -g1_config.joint_max_torques, g1_config.joint_max_torques)
    data.ctrl[:] = torque


def main():
    # Configuration values
    physics_freq = 1000
    policy_freq = 50
    decimation = int(physics_freq / policy_freq)
    policy_dt = 1 / policy_freq
    real_time = True

    # Motion tracker
    functional_tracker = FunctionalMotionTracker()
    motion_tracker = StatefulMotionTracker(functional_tracker)

    # Set up the sim
    model, data = initialize_environment()

    # Values for holding a standing position
    default_pos = INITIAL_QPOS[:3]
    default_quat = INITIAL_QPOS[3:7]
    default_vel = np.zeros(3)
    default_omega = np.zeros(3)
    default_q = INITIAL_QPOS[7:]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            pd_target = motion_tracker.step(
                robot_quat_wxyz=data.qpos[3:7],
                robot_omega_body=data.qvel[3:6],
                robot_q=data.qpos[-29:],
                robot_qd=data.qvel[-29:],
                mimic_pos=default_pos,
                mimic_quat=default_quat,
                mimic_vel=default_vel,
                mimic_omega=default_omega,
                mimic_q=default_q,
            )
            for _ in range(decimation):
                apply_pd_control(data, pd_target)
                mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - start_time
            if real_time and elapsed < policy_dt:
                time.sleep(policy_dt - elapsed)


if __name__ == "__main__":
    main()
