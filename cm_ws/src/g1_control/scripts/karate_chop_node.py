#!/usr/bin/env python3
"""
Karate chop demo node

This is essentially a ROS2 version of the demo from karate_chop.py,
deployable on the robot or in sim

State machine:
    WAIT_FOR_ROBOT -> IDLE -(advance)-> MOVE_TO_DEFAULT -> HOLD_DEFAULT
    -(advance)-> TRACK_STAND -(advance)-> RAISE_ARM -> HOLD_RAISED
    -(advance)-> RUN_MOTION -> LOWER_ARM -> TRACK_STAND (replay, or stop)

Inputs (either via terminal or by pressing the unitree remote buttons):
- Unitree remote (via /lowstate): A = advance, B = stop (hold the current
  commanded position), SELECT = emergency stop (robot goes limp)
- Topics:
    ros2 topic pub --once /g1_control/karate_chop/advance std_msgs/msg/Empty
    ros2 topic pub --once /g1_control/karate_chop/stop std_msgs/msg/Empty
    ros2 topic pub --once /g1_control/emergency_stop std_msgs/msg/Empty
"""

import signal
from enum import Enum
from pathlib import Path

import numpy as np
import jax

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
from g1_control_msgs.msg import RobotState, RobotCommand, FrameState

import cm_control
from cbfpy import CBF
from frax import load_g1
from cm_control.config import g1_config
from cm_control.core.actuators import RobotActuatorModel
from cm_control.core.dynamic_filter_utils import PDFilter
from cm_control.examples.karate_chop import (
    KarateChopKinematicCBFConfig,
    KarateChopDynamicCBFConfig,
)
from cm_control.core.simple_state_estimation import (
    FunctionalEstimator,
    StatefulEstimator,
)
from cm_control.retargeting.motion_processing import load_and_retarget_motion
from cm_control.retargeting.pico_to_g1_retarget import (
    PicoToG1Retargeter,
    StatefulRetargeter,
)
from cm_control.twist2_utils import twist2_config
from cm_control.twist2_utils.motion_tracking import (
    FunctionalMotionTracker,
    StatefulMotionTracker,
)
from cm_control.utils.rotation_utils import slerp

jax.config.update("jax_enable_x64", True)

DEFAULT_MOTION_FILE = str(
    Path(cm_control.__file__).parents[2]
    / "cm_control"
    / "cm_control"
    / "assets"
    / "data"
    / "rosbag2_2026_05_06-13_49_25_crop_2046_to_2450.pkl"
)

# Unitree remote bit indices. See unitree_ros2 gamepad.hpp
BUTTON_SELECT_BIT = 3
BUTTON_A_BIT = 8
BUTTON_B_BIT = 9


class State(Enum):
    WAIT_FOR_ROBOT = 0
    IDLE = 1
    MOVE_TO_DEFAULT = 2
    HOLD_DEFAULT = 3
    TRACK_STAND = 4
    RAISE_ARM = 5
    HOLD_RAISED = 6
    RUN_MOTION = 7
    LOWER_ARM = 8
    FINISHED = 9
    KILLED = 10


class KarateChopNode(Node):
    def __init__(self):
        super().__init__("karate_chop_node")

        # Tunable parameters
        self.declare_parameter("motion_file", DEFAULT_MOTION_FILE)
        self.declare_parameter("add_kinematic_cbf", False)
        self.declare_parameter("add_dynamic_cbf", False)
        # The root state can either be from 'topic', 'estimator', or 'mimic'
        # topic: use ground truth from sim or mocap on hardware
        # estimator: kinematic estimate assuming double-support contact
        # mimic: assume that the root state matches the reference motion
        self.declare_parameter("root_state_source", "estimator")
        self.declare_parameter("move_to_default_duration", 3.0)
        self.declare_parameter("transition_duration", 3.0)
        motion_file = self.get_parameter("motion_file").value
        add_kinematic_cbf = self.get_parameter("add_kinematic_cbf").value
        add_dynamic_cbf = self.get_parameter("add_dynamic_cbf").value
        self.root_state_source = self.get_parameter("root_state_source").value
        if self.root_state_source not in ("topic", "estimator", "mimic"):
            raise ValueError(f"Invalid root state source: {self.root_state_source}")
        self.move_to_default_duration = self.get_parameter(
            "move_to_default_duration"
        ).value
        self.transition_duration = self.get_parameter("transition_duration").value

        self.control_freq = 50
        self.control_dt = 1 / self.control_freq

        # PD constants and default (standing) reference
        self.stiffness = np.asarray(twist2_config.sim_kps)
        self.damping = np.asarray(twist2_config.sim_kds)
        self.default_q_actuated = np.asarray(twist2_config.default_joint_position)
        self.default_root_pos = np.array([0.0, 0.0, 0.8])
        self.default_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.zero_vec3 = np.zeros(3)
        self.default_contact_mode = 3  # Both feet

        # Safety filters (same configuration as the sim example)
        robot = load_g1()
        if add_kinematic_cbf:
            kin_cbf = CBF.from_config(KarateChopKinematicCBFConfig(robot))
        else:
            kin_cbf = None
        if add_dynamic_cbf:
            dyn_cbf = CBF.from_config(KarateChopDynamicCBFConfig(robot))
            actuator_model = RobotActuatorModel(
                self.stiffness, self.damping, g1_config.joint_max_torques
            )
            self.pd_filter = PDFilter(dyn_cbf, actuator_model)
        else:
            self.pd_filter = None
        if self.pd_filter is not None and self.root_state_source == "estimator":
            self.estimator = StatefulEstimator(FunctionalEstimator(robot))
        else:
            self.estimator = None

        # Load and retarget the reference motion.
        # Note: this demo doesn't run the constrained retargeter online, it instead
        # uses it to precompute the safe reference motion from human data
        inner_retargeter = PicoToG1Retargeter(
            kin_cbf, robot, use_constrained_integration=True, force_double_support=True
        )
        retargeter = StatefulRetargeter(inner_retargeter)
        self.get_logger().info(f"Loading and retargeting motion: {motion_file}")
        self.motion = load_and_retarget_motion(
            motion_file, self.control_freq, retargeter
        )
        self.motion_q_actuated = self.motion["q"][:, 6:]
        self.num_motion_steps = self.motion["q"].shape[0]

        self.motion_tracker = StatefulMotionTracker(FunctionalMotionTracker())
        if self.pd_filter is not None:
            self.get_logger().info("JIT compiling the PD filter...")
            if self.estimator is not None:
                jax.block_until_ready(
                    self.pd_filter.filter_from_virtual_joints(
                        self.default_q_actuated,
                        np.zeros(6),
                        np.zeros(6),
                        self.default_q_actuated,
                        np.zeros(29),
                        self.default_contact_mode,
                    )
                )
            else:
                jax.block_until_ready(
                    self.pd_filter.filter(
                        self.default_q_actuated,
                        self.default_root_pos,
                        self.default_root_quat,
                        self.zero_vec3,
                        self.zero_vec3,
                        self.default_q_actuated,
                        np.zeros(29),
                        self.default_contact_mode,
                    )
                )

        # State machine management
        self.state = State.WAIT_FOR_ROBOT
        self.advance_requested = False
        self.stop_requested = False
        self.estop_requested = False
        self.interp_start_time = None
        self.interp_start_q = None
        self.motion_step = 0
        self.hold_pd_target = None

        # Latest message caches
        self.last_robot_state = None
        self.last_root_state = None
        self.prev_remote_keys = 0

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
        self.root_state_sub = self.create_subscription(
            FrameState,
            "/g1_control/root_state",
            self.root_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.low_state_sub = self.create_subscription(
            LowState,
            "/lowstate",
            self.low_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.advance_sub = self.create_subscription(
            Empty,
            "/g1_control/karate_chop/advance",
            self.advance_callback,
            qos_reliable_keep_last_volatile_depth_1,
        )
        self.stop_sub = self.create_subscription(
            Empty,
            "/g1_control/karate_chop/stop",
            self.stop_callback,
            qos_reliable_keep_last_volatile_depth_1,
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
        self.estop_pub = self.create_publisher(
            Empty, "/g1_control/emergency_stop", qos_reliable_keep_last_volatile_depth_1
        )
        qos_reliable_keep_last_transient_depth_1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Virtual gantry is only for simulation
        self.gantry_pub = self.create_publisher(
            Bool, "/g1_control/sim_gantry", qos_reliable_keep_last_transient_depth_1
        )
        # Keep the robot supported until the policy takes over
        self.set_gantry(True)

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
        self.get_logger().info("Karate chop node initialized")
        self.get_logger().info(f"{add_kinematic_cbf=}")
        self.get_logger().info(f"{add_dynamic_cbf=}")
        self.get_logger().info(f"root_state_source={self.root_state_source}")
        self.get_logger().info("Controls: A = advance, B = stop (hold), SELECT = emergency stop")
        self.get_logger().info("(or publish to /g1_control/karate_chop/{advance,stop})")
        self.get_logger().info("====================================================")
        # fmt: on

    def robot_state_callback(self, msg):
        self.last_robot_state = msg

    def root_state_callback(self, msg):
        self.last_root_state = msg

    def low_state_callback(self, msg):
        # Parse the unitree remote button data
        keys = int(msg.wireless_remote[2]) | (int(msg.wireless_remote[3]) << 8)
        new_presses = keys & ~self.prev_remote_keys
        self.prev_remote_keys = keys
        if new_presses & (1 << BUTTON_A_BIT):
            self.advance_requested = True
        if new_presses & (1 << BUTTON_B_BIT):
            self.stop_requested = True
        if new_presses & (1 << BUTTON_SELECT_BIT):
            self.estop_requested = True

    def advance_callback(self, msg):
        self.advance_requested = True

    def stop_callback(self, msg):
        self.stop_requested = True

    def reset_callback(self, msg):
        # Restart the state machine from scratch
        self.get_logger().info("Reset received: restarting the state machine")
        self.state = State.WAIT_FOR_ROBOT
        self.advance_requested = False
        self.stop_requested = False
        self.estop_requested = False
        self.motion_step = 0
        self.hold_pd_target = None
        self.last_robot_state = None
        functional_tracker = self.motion_tracker.functional_tracker
        self.motion_tracker._last_action = np.zeros(functional_tracker.num_actions)
        self.motion_tracker._obs_history = np.zeros(
            (functional_tracker.history_len, functional_tracker.n_obs_single)
        )
        if self.estimator is not None:
            self.estimator.reset()
        self.set_gantry(True)

    def set_gantry(self, enabled):
        self.gantry_pub.publish(Bool(data=enabled))

    def get_current_time_in_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def interpolation_pct(self, duration):
        elapsed = self.get_current_time_in_seconds() - self.interp_start_time
        return min(elapsed / duration, 1.0)

    def publish_pd_command(self, q_target):
        q_target = np.asarray(q_target)
        msg = RobotCommand()
        msg.q = q_target.tolist()
        msg.dq = [0.0] * 29
        msg.kp = self.stiffness.tolist()
        msg.kd = self.damping.tolist()
        msg.tau = [0.0] * 29
        self.robot_command_pub.publish(msg)
        self.hold_pd_target = q_target

    def step_policy(
        self,
        mimic_q,
        mimic_pos,
        mimic_quat,
        mimic_vel,
        mimic_omega,
        contact_mode,
        apply_filter=False,
    ):
        robot_q = np.array(self.last_robot_state.q)
        robot_qd = np.array(self.last_robot_state.dq)
        robot_quat_wxyz = np.array(self.last_robot_state.quaternion)
        robot_omega_body = np.array(self.last_robot_state.gyroscope)
        # Keep the estimator's state up to date at every policy step, even
        # when the filter is not applied
        if self.estimator is not None:
            q_root, qd_root = self.estimator.step(robot_q, robot_qd)
        pd_target = self.motion_tracker.step(
            robot_quat_wxyz=robot_quat_wxyz,
            robot_omega_body=robot_omega_body,
            robot_q=robot_q,
            robot_qd=robot_qd,
            mimic_pos=mimic_pos,
            mimic_quat=mimic_quat,
            mimic_vel=mimic_vel,
            mimic_omega=mimic_omega,
            mimic_q=mimic_q,
        )
        # Note: (if the dynamic filter is enabled,) this runs the dynamic filter
        # at the same rate as the policy (50 Hz). This is totally fine for this demo
        # but other demos may require a higher frequency and should be decoupled
        if self.pd_filter is not None and apply_filter:
            if self.root_state_source == "estimator":
                pd_target = self.pd_filter.filter_from_virtual_joints(
                    pd_target, q_root, qd_root, robot_q, robot_qd, contact_mode
                )
            else:
                if self.root_state_source == "mimic":
                    pos, quat = mimic_pos, mimic_quat
                    vel, omega = mimic_vel, mimic_omega
                else:  # topic
                    pos = np.array(self.last_root_state.position)
                    quat = np.array(self.last_root_state.quaternion)
                    vel = np.array(self.last_root_state.velocity)
                    omega = np.array(self.last_root_state.omega)
                pd_target = self.pd_filter.filter(
                    pd_target, pos, quat, vel, omega, robot_q, robot_qd, contact_mode
                )
            # HACK -- update the last action state after filtering
            # (same as the karate chop sim example)
            self.motion_tracker._last_action = (
                1 / self.motion_tracker.functional_tracker.action_scale
            ) * (pd_target - self.motion_tracker.functional_tracker.default_dof_pos)
        self.publish_pd_command(pd_target)

    def track_interpolated_reference(self, start_step, end_step, pct):
        q0, pos0, quat0 = start_step
        q1, pos1, quat1 = end_step
        q = linear_interp(q0, q1, pct)
        pos = linear_interp(pos0, pos1, pct)
        quat = np.asarray(slerp(quat0, quat1, pct))  # TODO: use numpy slerp!
        self.step_policy(
            q, pos, quat, self.zero_vec3, self.zero_vec3, self.default_contact_mode
        )

    @property
    def stand_step(self):
        return (self.default_q_actuated, self.default_root_pos, self.default_root_quat)

    def motion_keyframe(self, i):
        return (
            self.motion_q_actuated[i],
            self.motion["pos"][i],
            self.motion["quat"][i],
        )

    def control_loop(self):
        # Consume any pending operator inputs
        advance, self.advance_requested = self.advance_requested, False
        stop, self.stop_requested = self.stop_requested, False
        estop, self.estop_requested = self.estop_requested, False

        # Handle a kill command immediately
        if estop and self.state != State.KILLED:
            self.get_logger().warn("Emergency stop requested! Killing robot")
            self.estop_pub.publish(Empty())
            self.state = State.KILLED
            # In sim, catch the limp robot on the virtual gantry
            self.set_gantry(True)
        if self.state == State.KILLED:
            return

        if self.last_robot_state is None:
            self.get_logger().info(
                "Waiting for robot state...", throttle_duration_sec=1.0
            )
            return
        if self.state == State.WAIT_FOR_ROBOT:
            self.state = State.IDLE
            self.get_logger().info(
                "Robot state received. Press A to move to the default pose"
            )
            return

        # A stop request (short of an estop) freezes the robot at the last
        # commanded position, with the policy turned off
        if stop and self.hold_pd_target is not None and self.state != State.FINISHED:
            self.get_logger().info(
                "Stop requested: holding the last commanded position"
            )
            self.state = State.FINISHED
            # The position hold is not actively balanced, so re-engage the
            # (virtual or real) gantry support
            self.set_gantry(True)

        if self.state == State.IDLE:
            if advance:
                self.interp_start_q = np.array(self.last_robot_state.q)
                self.interp_start_time = self.get_current_time_in_seconds()
                self.state = State.MOVE_TO_DEFAULT
                self.get_logger().info("Moving to the default pose...")
        elif self.state == State.MOVE_TO_DEFAULT:
            pct = self.interpolation_pct(self.move_to_default_duration)
            self.publish_pd_command(
                linear_interp(self.interp_start_q, self.default_q_actuated, pct)
            )
            if pct >= 1.0:
                self.state = State.HOLD_DEFAULT
                self.get_logger().info(
                    "Holding the default pose. Press A to activate the policy"
                )
        elif self.state == State.HOLD_DEFAULT:
            self.publish_pd_command(self.default_q_actuated)
            if advance:
                self.state = State.TRACK_STAND
                # The policy balances the robot from here on
                self.set_gantry(False)
                self.get_logger().info(
                    "Policy active (standing). Press A to raise the arm"
                )
        elif self.state == State.TRACK_STAND:
            self.track_interpolated_reference(self.stand_step, self.stand_step, 0.0)
            if advance:
                self.interp_start_time = self.get_current_time_in_seconds()
                self.state = State.RAISE_ARM
                self.get_logger().info("Raising the arm to the start of the motion...")
        elif self.state == State.RAISE_ARM:
            pct = self.interpolation_pct(self.transition_duration)
            self.track_interpolated_reference(
                self.stand_step, self.motion_keyframe(0), pct
            )
            if pct >= 1.0:
                self.state = State.HOLD_RAISED
                self.get_logger().info("Arm raised. Press A to run the motion")
        elif self.state == State.HOLD_RAISED:
            self.track_interpolated_reference(
                self.motion_keyframe(0), self.motion_keyframe(0), 0.0
            )
            if advance:
                if (
                    self.pd_filter is not None
                    and self.root_state_source == "topic"
                    and self.last_root_state is None
                ):
                    self.get_logger().warn(
                        "No root state received yet: the dynamic CBF needs "
                        "/g1_control/root_state (or another root_state_source)"
                    )
                else:
                    self.motion_step = 0
                    self.state = State.RUN_MOTION
                    self.get_logger().info("Running the reference motion...")
        elif self.state == State.RUN_MOTION:
            i = self.motion_step
            self.step_policy(
                self.motion_q_actuated[i],
                self.motion["pos"][i],
                self.motion["quat"][i],
                self.motion["vel"][i],
                self.motion["omega"][i],
                int(self.motion["contact_mode"][i]),
                apply_filter=True,
            )
            self.motion_step += 1
            if self.motion_step >= self.num_motion_steps:
                self.interp_start_time = self.get_current_time_in_seconds()
                self.state = State.LOWER_ARM
                self.get_logger().info("Motion complete. Returning to stand...")
        elif self.state == State.LOWER_ARM:
            pct = self.interpolation_pct(self.transition_duration)
            self.track_interpolated_reference(
                self.motion_keyframe(-1), self.stand_step, pct
            )
            if pct >= 1.0:
                self.state = State.TRACK_STAND
                self.get_logger().info(
                    "Standing. Press A to replay the motion, or B to stop and hold"
                )
        elif self.state == State.FINISHED:
            self.publish_pd_command(self.hold_pd_target)
        else:
            raise RuntimeError("Unreachable state")


def linear_interp(start, end, pct):
    return start + pct * (end - start)


# NOTE: upon trying a few methods, this seemed to be the best way to get the kill command
# sent out on a ctrl+c. But, sometimes there are still race condition issues, so maybe
# there is a better way to handle it (maybe with the C++ command processor node?)
def main(args=None):
    rclpy.init(args=args)
    node = KarateChopNode()

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
