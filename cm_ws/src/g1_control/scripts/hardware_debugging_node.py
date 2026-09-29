#!/usr/bin/env python3
"""
Hardware debugging node: move the ankle joint back and forth while
hanging on the gantry to debug if communication is working properly

SELECT on the unitree controller kills the robot (or ctrl+c)
"""

import signal

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import Bool, Empty
from unitree_hg.msg import LowState
from g1_control_msgs.msg import RobotState, RobotCommand

from cm_control.config import g1_config
from cm_control.twist2_utils import twist2_config


BUTTON_SELECT_BIT = 3  # See unitree_ros2 gamepad.hpp


class HardwareDebuggingNode(Node):
    def __init__(self):
        super().__init__("hardware_debugging_node")

        # Tunable parameters
        self.declare_parameter(
            "joint_index",
            g1_config.fixed_root_g1_joint_ordering.index("left_ankle_pitch_joint"),
        )
        self.declare_parameter("amplitude", 0.2)  # radians
        self.declare_parameter("period", 5.0)  # seconds
        self.joint_index = self.get_parameter("joint_index").value
        self.amplitude = self.get_parameter("amplitude").value
        self.period = self.get_parameter("period").value
        self.joint_name = g1_config.fixed_root_g1_joint_ordering[self.joint_index]

        self.control_freq = 50
        self.control_dt = 1 / self.control_freq

        self.stiffness = np.asarray(twist2_config.sim_kps)
        self.damping = np.asarray(twist2_config.sim_kds)

        # Set once the first robot state arrives
        self.start_time = None
        self.start_q = None

        # Latest message caches
        self.last_robot_state = None
        self.prev_remote_keys = 0
        self.killed = False

        # QoS settings
        qos_best_effort_keep_last_volatile_depth_1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_reliable_keep_last_volatile_depth_1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_reliable_keep_last_transient_depth_1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers and subscribers
        self.robot_state_sub = self.create_subscription(
            RobotState,
            "/g1_control/robot_state",
            self.robot_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.low_state_sub = self.create_subscription(
            LowState,
            "/lowstate",
            self.low_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.robot_command_pub = self.create_publisher(
            RobotCommand,
            "/g1_control/robot_command",
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.estop_pub = self.create_publisher(
            Empty, "/g1_control/emergency_stop", qos_reliable_keep_last_volatile_depth_1
        )
        # Publisher for the virtual gantry (sim only). Assumes hardware is already on gantry
        self.gantry_pub = self.create_publisher(
            Bool, "/g1_control/sim_gantry", qos_reliable_keep_last_transient_depth_1
        )
        self.gantry_pub.publish(Bool(data=True))

        # Keep a message prepared to send in an emergency kill state
        kill_msg = RobotCommand()
        kill_msg.q = [0.0] * 29
        kill_msg.dq = [0.0] * 29
        kill_msg.kp = [0.0] * 29
        kill_msg.kd = [1.0] * 29  # Slight damping
        kill_msg.tau = [0.0] * 29
        self.kill_msg = kill_msg

        self.timer = self.create_timer(self.control_dt, self.control_loop)

        # fmt: off
        self.get_logger().info("====================================================")
        self.get_logger().info("Hardware debugging node initialized")
        self.get_logger().info(f"Oscillating {self.joint_name} (index {self.joint_index})")
        self.get_logger().info(f"amplitude={self.amplitude} rad, period={self.period} s")
        self.get_logger().info("Holding the starting pose on all other joints. SELECT = emergency stop")
        self.get_logger().info("====================================================")
        # fmt: on

    def robot_state_callback(self, msg):
        self.last_robot_state = msg

    def low_state_callback(self, msg):
        # Parse the unitree remote button data, triggering on rising edges
        keys = int(msg.wireless_remote[2]) | (int(msg.wireless_remote[3]) << 8)
        new_presses = keys & ~self.prev_remote_keys
        self.prev_remote_keys = keys
        if new_presses & (1 << BUTTON_SELECT_BIT):
            self.get_logger().warn("Emergency stop requested! Killing robot")
            self.estop_pub.publish(Empty())
            self.killed = True

    def get_current_time_in_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def control_loop(self):
        if self.killed:
            return
        if self.last_robot_state is None:
            self.get_logger().info(
                "Waiting for robot state...", throttle_duration_sec=1.0
            )
            return
        if self.start_time is None:
            self.start_time = self.get_current_time_in_seconds()
            self.start_q = np.array(self.last_robot_state.q)
            self.get_logger().info(
                "Robot state received. Holding the starting pose and "
                f"oscillating about q = {self.start_q[self.joint_index]:.3f}"
            )
        # Hold the starting pose while oscillating the debug joint
        elapsed = self.get_current_time_in_seconds() - self.start_time
        q_target = self.start_q.copy()
        q_target[self.joint_index] += self.amplitude * np.sin(
            2 * np.pi * elapsed / self.period
        )
        msg = RobotCommand()
        msg.q = q_target.tolist()
        msg.dq = [0.0] * 29
        msg.kp = self.stiffness.tolist()
        msg.kd = self.damping.tolist()
        msg.tau = [0.0] * 29
        self.robot_command_pub.publish(msg)
        self.get_logger().info(
            f"commanded = {q_target[self.joint_index]:.3f}, "
            f"measured = {self.last_robot_state.q[self.joint_index]:.3f}",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = HardwareDebuggingNode()

    shutting_down = False

    def sigint_handler(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True

        node.get_logger().warn("SIGINT received, killing robot")

        for _ in range(3):
            node.robot_command_pub.publish(node.kill_msg)
            rclpy.spin_once(node, timeout_sec=0.05)

        rclpy.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()


if __name__ == "__main__":
    main()
