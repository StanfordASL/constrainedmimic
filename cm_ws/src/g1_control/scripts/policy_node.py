#!/usr/bin/env python3
"""
Policy node

This node calls the motion tracking policy (given the robot state
and reference motion to track) and returns the joint PD targets
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import signal

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import Empty
from g1_control_msgs.msg import RobotState, RobotCommand, MimicObservation

from cm_control.twist2_utils.jax_twist2 import load_policy
from cm_control.twist2_utils.twist2_utils import (
    build_proprio_obs,
    build_full_obs_and_new_history,
)
from cm_control.twist2_utils import twist2_config

APPLY_FILTER = False

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_log_compiles", True)


@jax.tree_util.register_static
class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")

        # Load JAX policy
        self.get_logger().info("Loading actor model...")
        self.nn_model, self.nn_params = load_policy()

        # G1 Specific Constants (JAX Arrays)
        self.num_actions = 29
        self.action_scale = 0.5
        self.default_dof_pos = np.asarray(twist2_config.default_joint_position)
        # PD Constants
        self.stiffness = np.asarray(twist2_config.sim_kps)
        self.damping = np.asarray(twist2_config.sim_kds)

        self.n_mimic_obs = 35
        self.history_len = 10
        self.n_obs_single = 127

        # History buffers
        self.obs_history = np.zeros((self.history_len, self.n_obs_single))
        self.last_action = np.zeros(self.num_actions)

        self.last_robot_state = None
        self.last_mimic_obs = None

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

        # Publishers and subscribers
        self.robot_state_sub = self.create_subscription(
            RobotState,
            "/g1_control/robot_state",
            self.robot_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.mimic_obs_sub = self.create_subscription(
            MimicObservation,
            "/g1_control/mimic_obs",
            self.mimic_obs_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.reset_sub = self.create_subscription(
            Empty,
            "/g1_control/reset",
            self.reset_callback,
            qos_reliable_keep_last_volatile_depth_1,
        )
        self.robot_command_pub = self.create_publisher(
            RobotCommand,
            "/g1_control/robot_command",
            qos_best_effort_keep_last_volatile_depth_1,
        )

        # Keep a message prepared to send in an emergency kill state
        kill_msg = RobotCommand()
        kill_msg.q = [0.0] * 29
        kill_msg.dq = [0.0] * 29
        kill_msg.kp = [0.0] * 29
        kill_msg.kd = [1.0] * 29  # Slight damping
        kill_msg.tau = [0.0] * 29
        self.kill_msg = kill_msg

        self.control_freq = 50
        self.timer = self.create_timer(1 / self.control_freq, self.control_loop)

        self.get_logger().info("Jit compiling policy and obs logic...")
        self._jit_compile()
        self.get_logger().info("Jit compilation complete.")

    def _jit_compile(self):
        # Trigger JIT for policy and observation processing
        dummy_quat = np.array([1.0, 0.0, 0.0, 0.0])
        dummy_ang_vel = np.zeros(3)
        dummy_dof_pos = self.default_dof_pos
        dummy_dof_vel = np.zeros(29)
        dummy_mimic_obs = np.zeros(self.n_mimic_obs)
        history = np.zeros((self.history_len, self.n_obs_single))
        last_action = np.zeros(self.num_actions)

        num_warmup = self.history_len
        for _ in range(num_warmup):
            pd_target, last_action, history = self._step_policy(
                dummy_quat,
                dummy_ang_vel,
                dummy_dof_pos,
                dummy_dof_vel,
                dummy_mimic_obs,
                history,
                last_action,
            )
            pd_target.block_until_ready()
            last_action.block_until_ready()
            history.block_until_ready()

    def reset_callback(self, msg):
        self.get_logger().info("Resetting JAX control node history...")
        self.obs_history = np.zeros((self.history_len, self.n_obs_single))
        self.last_action = np.zeros(self.num_actions)
        self.last_mimic_obs = None
        self.last_contact_mode = 3

    def robot_state_callback(self, msg):
        self.last_robot_state = msg

    def mimic_obs_callback(self, msg):
        self.last_mimic_obs = msg.data

    @partial(jax.jit, static_argnums=(0,))
    def _step_policy(
        self,
        robot_quat_wxyz,
        robot_omega_body,
        robot_q,
        robot_qd,
        mimic_obs,
        history,
        last_action,
    ):
        # TWIST2 logic and policy inputs
        proprio_obs = build_proprio_obs(
            robot_quat_wxyz, robot_omega_body, robot_q, robot_qd, last_action
        )
        full_obs, new_history = build_full_obs_and_new_history(
            mimic_obs, proprio_obs, history
        )
        # Call TWIST2 policy
        raw_action = self.nn_model.apply(
            self.nn_params, full_obs[jnp.newaxis, :], train=False
        ).squeeze()
        # Compute PD target
        pd_target = raw_action * self.action_scale + self.default_dof_pos
        return pd_target, raw_action, new_history

    # No jit
    def control_loop(self):
        if self.last_robot_state is None or self.last_mimic_obs is None:
            self.get_logger().info(
                "No state or mimic data received yet", throttle_duration_sec=1.0
            )
            return

        robot_q = np.array(self.last_robot_state.q)
        robot_qd = np.array(self.last_robot_state.dq)
        robot_quat_wxyz = np.array(self.last_robot_state.quaternion)
        robot_omega_body = np.array(self.last_robot_state.gyroscope)

        pd_target, self.last_action, self.obs_history = self._step_policy(
            robot_quat_wxyz,
            robot_omega_body,
            robot_q,
            robot_qd,
            np.asarray(self.last_mimic_obs),
            self.obs_history,
            self.last_action,
        )

        cmd_msg = RobotCommand()
        cmd_msg.q = pd_target.tolist()
        cmd_msg.dq = [0.0] * 29
        cmd_msg.kp = self.stiffness.tolist()
        cmd_msg.kd = self.damping.tolist()
        cmd_msg.tau = [0.0] * 29
        self.robot_command_pub.publish(cmd_msg)


# NOTE: upon trying a few methods, this seemed to be the best way to get the kill command
# sent out on a ctrl+c. But, sometimes there are still race condition issues, so maybe
# there is a better way to handle it (maybe with the C++ command processor node?)
def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()

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
