import os
import mujoco
import mujoco.viewer
import numpy as np
import jax.numpy as jnp
import xml.etree.ElementTree as ET
import copy
from frax import load_g1
from cbfpy import CBF
from cm_control.assets import G1_XML
from cm_control.utils.rotation_utils import quat_wxyz_to_intrinsic_euler_xyz


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

    return root


def add_visual_frames(root, num_frames):
    worldbody = root.find("worldbody")
    for i in range(num_frames):
        # Create a mocap body for each frame
        frame_body = ET.SubElement(
            worldbody, "body", name=f"frame_{i}", mocap="true", pos="0 0 0"
        )
        # X axis (Red)
        ET.SubElement(
            frame_body,
            "geom",
            type="cylinder",
            size="0.005 0.05",
            rgba="1 0 0 0.8",
            pos="0.05 0 0",
            quat="0.7071 0 0.7071 0",
            group="1",
        )
        # Y axis (Green)
        ET.SubElement(
            frame_body,
            "geom",
            type="cylinder",
            size="0.005 0.05",
            rgba="0 1 0 0.8",
            pos="0 0.05 0",
            quat="0.7071 -0.7071 0 0",
            group="1",
        )
        # Z axis (Blue)
        ET.SubElement(
            frame_body,
            "geom",
            type="cylinder",
            size="0.005 0.05",
            rgba="0 0 1 0.8",
            pos="0 0 0.05",
            quat="1 0 0 0",
            group="1",
        )


def mat2quat(R):
    """Simple 3x3 rotation matrix to quaternion (WXYZ) conversion for visualization."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        # Simplified for visualization, could be improved for robust cases
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def main():
    # Note that my floating root model has 35 dof:
    # 6dof PPPRRR joints to model the freefloating root
    # plus the 29 actuated joints
    robot = load_g1()
    # q = np.zeros(robot.num_joints)
    # link_tfs = robot.link_to_world_transforms(q)

    # Please visualize the frames in mujoco on top of a "ghost"
    # unitree g1 with it loaded at the origin and all joints set to 0
    xml_path = str(G1_XML)
    root = ghostify_model(xml_path)
    add_visual_frames(root, robot.num_joints)
    xml_string = ET.tostring(root, encoding="unicode")

    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Set q to MuJoCo default robot pose (which includes the freejoint at pelvis)
    # The first 7 elements of data.qpos are [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    # We need to map this back to our 6DOF Euler root + 29 joints
    q_mujo = data.qpos[: 7 + 29]
    pos = q_mujo[:3]
    quat = q_mujo[3:7]
    joints = q_mujo[7:]

    euler = quat_wxyz_to_intrinsic_euler_xyz(quat)
    q = np.concatenate([pos, euler, joints])

    # Re-calculate link transforms with the updated q
    link_tfs = robot.link_to_world_transforms(q)

    # Set ghost robot to zero pose (already there by default in qpos)
    # We just need to make sure the mocap bodies are positioned correctly
    for i in range(robot.num_joints):
        tf = link_tfs[i]
        pos = tf[:3, 3]
        rot = tf[:3, :3]
        quat = mat2quat(rot)

        # Mocap bodies are at the end of the body list usually
        # but let's find them by name
        mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"frame_{i}")
        mocap_idx = model.body_mocapid[mocap_id]

        data.mocap_pos[mocap_idx] = pos
        data.mocap_quat[mocap_idx] = quat

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera to initial position
        viewer.cam.lookat = [0.0, 0.0, 0.7]
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
