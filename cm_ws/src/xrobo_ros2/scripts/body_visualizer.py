#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import threading
from xrobo_ros2.msg import XRoboState
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from scipy.spatial.transform import Rotation


def Rx(theta: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def create_mjcf(num_bodies):
    xml = f"""
    <mujoco>
        <option timestep="0.01"/>
        <visual>
            <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
        </visual>
        <worldbody>
    """
    for i in range(num_bodies):
        # 0: Headset
        # 1-24: Body
        # 25-26: Controllers (Left, Right)
        rgba = "1 1 1 1"
        if i == 0:
            rgba = "1 1 0 1"  # Yellow for headset
        elif i <= 24:
            rgba = "0 1 1 1"  # Cyan for body
        else:
            rgba = "1 0.5 0 1"  # Orange for controllers

        xml += f"""
            <body name="b_{i}" mocap="true">
                <geom type="sphere" size="0.02" rgba="{rgba}"/>
                <geom type="capsule" size="0.004" fromto="0 0 0 0.1 0 0" rgba="1 0 0 1"/>
                <geom type="capsule" size="0.004" fromto="0 0 0 0 0.1 0" rgba="0 1 0 1"/>
                <geom type="capsule" size="0.004" fromto="0 0 0 0 0 0.1" rgba="0 0 1 1"/>
            </body>
        """
    xml += """
        </worldbody>
    </mujoco>
    """
    return xml


class BodyVisualizer(Node):
    def __init__(self, adjust_arm_rotations: bool = False):
        super().__init__("body_visualizer")

        self.body_names = [
            "Pelvis",
            "Left_Hip",
            "Right_Hip",
            "Spine1",
            "Left_Knee",
            "Right_Knee",
            "Spine2",
            "Left_Ankle",
            "Right_Ankle",
            "Spine3",
            "Left_Foot",
            "Right_Foot",
            "Neck",
            "Left_Collar",
            "Right_Collar",
            "Head",
            "Left_Shoulder",
            "Right_Shoulder",
            "Left_Elbow",
            "Right_Elbow",
            "Left_Wrist",
            "Right_Wrist",
            "Left_Hand",
            "Right_Hand",
        ]
        self.body_name_to_idx = {name: idx for idx, name in enumerate(self.body_names)}
        self.adjust_arm_rotations = adjust_arm_rotations
        if self.adjust_arm_rotations:
            # fmt: off
            self.rotation_adjustments = {}
            self.rotation_adjustments[self.body_name_to_idx["Left_Shoulder"]] = Rx(np.deg2rad(90))
            self.rotation_adjustments[self.body_name_to_idx["Left_Elbow"]] = Ry(np.deg2rad(90)) @ Rz(np.deg2rad(90))
            self.rotation_adjustments[self.body_name_to_idx["Left_Wrist"]] = Ry(np.deg2rad(90)) @ Rz(np.deg2rad(90))
            self.rotation_adjustments[self.body_name_to_idx["Left_Hand"]] = Ry(np.deg2rad(90)) @ Rz(np.deg2rad(90))
            self.rotation_adjustments[self.body_name_to_idx["Right_Shoulder"]] = Rx(np.deg2rad(-90))
            self.rotation_adjustments[self.body_name_to_idx["Right_Elbow"]] = Rz(np.deg2rad(-90)) @ Rx(np.deg2rad(-90))
            self.rotation_adjustments[self.body_name_to_idx["Right_Wrist"]] = Rz(np.deg2rad(-90)) @ Rx(np.deg2rad(-90))
            self.rotation_adjustments[self.body_name_to_idx["Right_Hand"]] = Rz(np.deg2rad(-90)) @ Rx(np.deg2rad(-90))
            # fmt: on
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # 0: Headset
        # 1-24: Body
        # 25-26: Controllers
        self.num_bodies = 1 + 24 + 2
        self.mj_xml = create_mjcf(self.num_bodies)
        self.model = mujoco.MjModel.from_xml_string(self.mj_xml)
        self.data = mujoco.MjData(self.model)

        # Initialize mocap positions
        self.data.mocap_pos[:] = 0.0

        self.sub = self.create_subscription(
            XRoboState, "xrobo/state", self.state_cb, qos_profile
        )

        self.data_lock = threading.Lock()

        # Start MuJoCo viewer
        self.get_logger().info("Launching MuJoCo viewer...")
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )

        # Look at the front of the body
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -15
        self.viewer.cam.lookat = [0, 0, -1.0]

        self.timer = self.create_timer(0.02, self.update_viewer)
        self.get_logger().info("Body Visualizer ready")

    def state_cb(self, msg):
        with self.data_lock:
            # 0: Headset
            self.data.mocap_pos[0] = msg.headset_pose[:3]
            self.data.mocap_quat[0] = msg.headset_pose[3:]

            # 1-24: Body
            for i in range(24):
                idx = 1 + i
                pos_start = i * 7
                self.data.mocap_pos[idx] = msg.flat_body_poses[
                    pos_start : pos_start + 3
                ]
                quat_wxyz = msg.flat_body_poses[pos_start + 3 : pos_start + 7]

                if self.adjust_arm_rotations and i in self.rotation_adjustments:
                    # Note: these adjustments happen in body frame, not world
                    # Note: scipy uses xyzw quats
                    R = self.rotation_adjustments[i]
                    R_adj_scipy = Rotation.from_matrix(R)
                    R_q_scipy = Rotation.from_quat(
                        np.asarray(quat_wxyz)[np.array([1, 2, 3, 0])]
                    )
                    q_scipy_new = (R_q_scipy * R_adj_scipy).as_quat()
                    quat_wxyz = q_scipy_new[np.array([3, 0, 1, 2])].tolist()
                self.data.mocap_quat[idx] = quat_wxyz

            # 25: Left Controller
            self.data.mocap_pos[25] = msg.left_controller_pose[:3]
            self.data.mocap_quat[25] = msg.left_controller_pose[3:]

            # 26: Right Controller
            self.data.mocap_pos[26] = msg.right_controller_pose[:3]
            self.data.mocap_quat[26] = msg.right_controller_pose[3:]

    def update_viewer(self):
        with self.data_lock:
            self.data.time += 0.02
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()


def main():
    rclpy.init()
    node = BodyVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
