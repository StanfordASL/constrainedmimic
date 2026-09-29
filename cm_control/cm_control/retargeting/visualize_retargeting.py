import os
import mujoco
import mujoco.viewer
import numpy as np
import jax.numpy as jnp
import xml.etree.ElementTree as ET
import copy
from cm_control.retargeting.retarget import KinematicRetargeter
from cm_control.retargeting.retargeting_cbf_configs import (
    JointLimitsFullyActuatedConfig,
)
from frax import load_g1
from cm_control.config.retargeting_config import G1_RETARGETING_IDXS
from cbfpy import CBF
from cm_control.assets import G1_XML


def ghostify_model(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # Prefix for ghost components
    GHOST_PREFIX = "ghost_"

    # Fix meshdir to be absolute so MuJoCo can find meshes when loading from string
    compiler = root.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        meshdir = compiler.attrib["meshdir"]
        if not os.path.isabs(meshdir):
            compiler.attrib["meshdir"] = str(
                os.path.abspath(os.path.join(os.path.dirname(xml_path), meshdir))
            )

    # We want to duplicate all top-level bodies that are part of the robot
    # In G1 MJCF, this is just the 'pelvis' body
    robot_body = worldbody.find("body[@name='pelvis']")
    if robot_body is None:
        raise ValueError("Could not find 'pelvis' body in MJCF")

    ghost_body = copy.deepcopy(robot_body)

    def prefix_recursive(node):
        if "name" in node.attrib:
            node.attrib["name"] = GHOST_PREFIX + node.attrib["name"]
        if "joint" in node.attrib:
            node.attrib["joint"] = GHOST_PREFIX + node.attrib["joint"]
        if node.tag == "geom":
            # Set transparency and group
            node.attrib["rgba"] = "0.2 0.5 0.8 0.3"
            node.attrib["group"] = "1"
        for child in node:
            prefix_recursive(child)

    prefix_recursive(ghost_body)
    worldbody.append(ghost_body)

    return ET.tostring(root, encoding="unicode")


def q_humanoid_to_mujoco(q, model, prefix=""):
    """Converts Humanoid 6D float root + joints to MuJoCo 7D freejoint + joints."""
    # Freejoint is at index 0 for each robot usually
    # But let's find it by name to be safe
    joint_name = prefix + "floating_base_joint"
    try:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    except:
        # If URDF root joint naming is different
        joint_name = prefix + "root"
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    qpos_adr = model.jnt_qposadr[jnt_id]

    # Root pos
    q_mj = np.zeros(
        model.nq
    )  # We only fill the slice we care about later or return a slice?
    # Better to just return the slice values

    pos = q[:3]
    from cm_control.utils.rotation_utils import intrinsic_euler_xyz_to_quat_wxyz

    quat = intrinsic_euler_xyz_to_quat_wxyz(q[3:6])

    res = np.zeros(7 + (q.shape[0] - 6))
    res[:3] = pos
    res[3:7] = quat
    res[7:] = q[6:]
    return res, qpos_adr


def main():
    # 1. Load Robot and Retargeter
    robot = load_g1()
    cbf_config = JointLimitsFullyActuatedConfig(robot)
    cbf = CBF.from_config(cbf_config)
    idxs = jnp.asarray(G1_RETARGETING_IDXS)

    kp_pos = 1.0
    kp_rot = 1.0
    weight_pos = 1.0
    weight_rot = 0.1
    stepsize = 1.0
    convergence_tol = 1e-3
    max_iters = 1  # We'll step it manually in the loop
    retargeter = KinematicRetargeter(
        cbf,
        robot,
        idxs,
        kp_pos,
        kp_rot,
        weight_pos,
        weight_rot,
        stepsize,
        convergence_tol,
        max_iters,
    )

    # 2. MuJoCo Setup with Ghost
    xml_path = str(G1_XML)
    ghost_xml = ghostify_model(xml_path)
    model = mujoco.MjModel.from_xml_string(ghost_xml)
    data = mujoco.MjData(model)

    # 3. Create Target Configuration
    np.random.seed(0)
    q_init = np.zeros(robot.num_joints)
    q_init[2] = 1.0  # Raise it up
    # Define a goal that is somewhat different but feasible
    q_goal = q_init.copy()
    # Add some random perturbation to joints
    q_goal[6:] += np.random.uniform(-0.5, 0.5, robot.num_actuated_joints)

    # Forward kinematics for goal
    all_link_tfs = robot.link_to_world_transforms(q_goal)
    selected_link_positions = all_link_tfs[idxs, :3, 3]
    selected_link_rotations = all_link_tfs[idxs, :3, :3]

    # 4. Set Ghost Robot Pose (Constant)
    q_mj_goal, ghost_adr = q_humanoid_to_mujoco(q_goal, model, prefix="ghost_")
    data.qpos[ghost_adr : ghost_adr + len(q_mj_goal)] = q_mj_goal
    mujoco.mj_forward(model, data)

    # 5. Visualization Loop
    viewer = mujoco.viewer.launch_passive(model, data)

    q_current = q_init.copy()

    import time

    print("Starting visualization...")
    input("Press Enter to start...")

    try:
        i = 0
        while viewer.is_running():
            step_start = time.time()

            # Step retargeter (1 iter)
            q_current, converged, iters = retargeter.scp_retarget(
                q_current, selected_link_positions, selected_link_rotations
            )

            # Update Real Robot pose in MuJoCo
            q_mj_real, real_adr = q_humanoid_to_mujoco(q_current, model, prefix="")
            data.qpos[real_adr : real_adr + len(q_mj_real)] = q_mj_real

            # Compute task-space errors for logging
            all_link_tfs_out = robot.link_to_world_transforms(q_current)
            selected_link_positions_out = all_link_tfs_out[idxs, :3, 3]
            pos_err = jnp.linalg.norm(
                selected_link_positions_out - selected_link_positions, axis=-1
            )
            mean_pos_err = jnp.mean(pos_err)

            i += 1
            print(f"Iter {i}, Mean Task Pos Error: {mean_pos_err:.6f}")

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Use smaller sleep for visual smoothness but don't bottleneck testing
            time.sleep(0.01)

            if converged:
                print("Converged!")
                print(f"Number of iterations: {i}")
                input("Press Enter to exit...")
                break

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
