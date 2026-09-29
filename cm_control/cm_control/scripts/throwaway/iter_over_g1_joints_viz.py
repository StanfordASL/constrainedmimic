"""Iterating over the unitree g1 joints to visualize name/joint correspondence"""

import time

import numpy as np
import mujoco
import mujoco.viewer

from cm_control.assets import G1_XML


def visualize_joint_ranges(xml_path):
    xml_path = str(xml_path)
    # 1. Load the model
    print(f"Loading model from: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading XML: {e}")
        return

    # 2. Launch the passive viewer
    # We use launch_passive so we can control the flow with our own loops
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset to initial configuration (qpos0)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        viewer.sync()

        print("\n" + "=" * 50)
        print(f"Model loaded. Total Joints: {model.njnt}")
        print("Press Enter in the console to cycle through joints.")
        print("=" * 50 + "\n")

        # 3. Iterate over all joints
        for jnt_id in range(model.njnt):
            # Get joint name
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)

            # Get joint type (we only want to animate HINGE or SLIDE)
            j_type = model.jnt_type[jnt_id]

            # Skip free joints (floating base) or ball joints for this simple demo
            if j_type not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                print(f"Skipping Joint {jnt_id} ({name}): Not a hinge or slide joint.")
                continue

            # Get address in qpos array
            q_adr = model.jnt_qposadr[jnt_id]

            # Determine Limits
            is_limited = model.jnt_limited[jnt_id]
            if is_limited:
                min_lim, max_lim = model.jnt_range[jnt_id]
            else:
                # If continuous/unlimited, create an arbitrary range for visualization
                min_lim, max_lim = -3.14, 3.14
                print(f"Joint {name} is unlimited. Using arbitrary range [-pi, pi].")

            print(f"--> Testing Joint: {name} | Range: [{min_lim:.2f}, {max_lim:.2f}]")

            # Store the original position to restore later
            original_qpos = data.qpos[q_adr]

            # 4. Animate the joint
            # We sweep from min to max and back to min
            duration = 2.0  # seconds per sweep
            steps = 150

            # Generate a smooth sine wave pattern normalized between 0 and 1
            t_vals = np.linspace(0, np.pi * 2, steps)
            sine_wave = (1 - np.cos(t_vals)) / 2  # Goes 0 -> 1 -> 0

            for factor in sine_wave:
                if not viewer.is_running():
                    return

                # Interpolate position
                target_pos = min_lim + (max_lim - min_lim) * factor

                # Update strictly the joint position
                data.qpos[q_adr] = target_pos

                # Forward kinematics only (computes positions/visuals, ignores physics/gravity)
                mujoco.mj_forward(model, data)

                viewer.sync()
                time.sleep(duration / steps)

            # Reset this joint to original position before moving to the next
            data.qpos[q_adr] = original_qpos
            mujoco.mj_forward(model, data)
            viewer.sync()

            # 5. Wait for user input
            input(f"    [Paused] Done with '{name}'. Press Enter to continue...")

    print("Done iterating through all joints.")


if __name__ == "__main__":
    visualize_joint_ranges(G1_XML)
