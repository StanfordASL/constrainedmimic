"""Interactive control of the G1 joints in the mujoco viewer

This will NOT apply physics and will allow for the joints to move in the viewer
"""

import mujoco
import mujoco.viewer
import time
import numpy as np

from cm_control.assets import G1_XML


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
INITIAL_JOINT_POS = INITIAL_QPOS[7:]


def main():
    xml_path = str(G1_XML)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    model.opt.gravity[:] = [0, 0, 0]

    joint_lower_limits, joint_upper_limits = model.jnt_range[-29:].T
    joint_ranges = joint_upper_limits - joint_lower_limits
    joint_midpoints = joint_lower_limits + joint_ranges / 2

    np.set_printoptions(suppress=True, threshold=10000, linewidth=400)

    left_leg_idx = 7
    right_leg_idx = left_leg_idx + 6
    torso_idx = right_leg_idx + 6
    left_arm_idx = torso_idx + 3
    right_arm_idx = left_arm_idx + 7

    dt_sim = model.opt.timestep
    real_time = True

    # Need to figure out: for our desired initial position,
    # what would this correspond to in the remapped control range?
    data.ctrl = (INITIAL_JOINT_POS - joint_midpoints) / (joint_ranges / 2)
    # Also update the data to match the root position
    data.qpos = INITIAL_QPOS

    waypoints = []

    def key_callback(keycode):
        """Callback to print the qpos to the screen on a spacebar press"""
        if chr(keycode) == " ":
            waypoints.append(np.copy(data.qpos))
            print(f"\n[SAVED] Waypoint {len(waypoints)} recorded!")

            # fmt: off
            print("np.array([")
            print(f"{np.array2string(data.qpos[:left_leg_idx], separator=", ")},  # Base")
            print(f"{np.array2string(data.qpos[left_leg_idx:right_leg_idx], separator=", ")},  # Left leg")
            print(f"{np.array2string(data.qpos[right_leg_idx:torso_idx], separator=", ")},  # Right leg")
            print(f"{np.array2string(data.qpos[torso_idx:left_arm_idx], separator=", ")},  # Torso")
            print(f"{np.array2string(data.qpos[left_arm_idx:right_arm_idx], separator=", ")},  # Left arm")
            print(f"{np.array2string(data.qpos[right_arm_idx:], separator=", ")},  # Right arm")
            print("])")
            # fmt: on

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        i = 0
        while viewer.is_running():
            start_time = time.time()

            # This is a bit of a hack
            # The sliders are from -1 to 1 and used to control the applied torque
            # BUT I want them to control the joint position instead
            ctrl_values = data.ctrl
            remapped_values = joint_midpoints + ctrl_values * joint_ranges / 2
            data.qpos[7:] = remapped_values
            # NOTE that the values in the sliders do NOT equal the joint positions
            # so we'll print that data out occasionally

            mujoco.mj_forward(model, data)
            viewer.sync()

            elapsed = time.time() - start_time
            if real_time and elapsed < dt_sim:
                time.sleep(dt_sim - elapsed)
            i += 1


if __name__ == "__main__":
    main()
