#!/usr/bin/env python3

from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from g1_control_msgs.msg import TeleopState
from std_msgs.msg import Empty

import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from cbfpy import CBF

from xrobo_ros2.msg import XRoboState

from cm_control.twist2_utils.twist2_config import default_joint_position
from cm_control.core.kinematic_configs import BaseKinematicConfig
from cm_control.retargeting.pico_to_g1_retarget import (
    PicoToG1Retargeter,
    StatefulRetargeter,
    RetargeterState,
)
from frax import load_g1


jax.config.update("jax_enable_x64", True)


class State(Enum):
    STAND = 0
    INTERP_STAND_TO_TELEOP = 1
    TELEOP = 2
    INTERP_TELEOP_TO_STAND = 3
    KILLED = 4


# class KarateChopKinematicCBFConfig(BaseKinematicConfig):
#     def __init__(self, robot, z_min):
#         self.z_min = z_min
#         super().__init__(
#             constrained=True,
#             underactuated=False,
#             robot=robot,
#             use_naive_objective=False,
#             solver_tol=1e-5,
#             init_args=None,  # Same as default
#             init_kwargs={"contact_mode": 3},
#         )

#     def h_1(self, z, *args, **kwargs):
#         q = z
#         collision_pos, collision_rad = self.robot.link_collision_data(q)
#         right_hand_index = 44
#         right_hand_pos = collision_pos[right_hand_index]
#         right_hand_rad = collision_rad[right_hand_index]
#         right_hand_z = right_hand_pos[2]
#         return jnp.array([right_hand_z - self.z_min - right_hand_rad])

#     def alpha(self, h, *args, **kwargs):
#         return 10.0 * h


# @jax.tree_util.register_static
class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")

        # Tunable parameters
        self.declare_parameter("use_kinematic_cbf", False)
        self.declare_parameter("stepsize", 0.2)
        self.declare_parameter("max_iters", 20)
        self.declare_parameter("convergence_tol", 1e-3)
        self.declare_parameter("publish_frequency", 100.0)

        use_kinematic_cbf = self.get_parameter("use_kinematic_cbf").value
        stepsize = self.get_parameter("stepsize").value
        max_iters = self.get_parameter("max_iters").value
        convergence_tol = self.get_parameter("convergence_tol").value
        publish_freq = self.get_parameter("publish_frequency").value
        self.dt_publish = 1.0 / publish_freq

        self.robot = load_g1()

        if use_kinematic_cbf:
            raise NotImplementedError(
                "TODO: Figure out a clean way to load a specified CBF (or hard-code it)"
            )
            # cbf_config = KarateChopKinematicCBFConfig(self.robot, z_min=0.75)
            # cbf = CBF.from_config(cbf_config)
        else:
            self.get_logger().info("NOT using kinematic CBF in teleop node")
            cbf = None

        inner_retargeter = PicoToG1Retargeter(
            cbf,
            self.robot,
            stepsize=stepsize,
            convergence_tol=convergence_tol,
            max_iters=max_iters,
        )
        self.retargeter = StatefulRetargeter(inner_retargeter)

        # fmt: off
        # TODO import this info from some config
        self.default_q_actuated = np.asarray(default_joint_position)
        # TODO!! Need to decide if default root height should be 0.779 or 0.8!
        # 0.8 is the twist2 value but 0.779 actually initializes in contact
        self.default_root_pos = np.array([0.0, 0.0, 0.8])
        self.default_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.default_euler = np.zeros(3)
        self.default_velocity = np.zeros(3)
        self.default_contact_mode = 3  # Both feet
        self.default_q_full = np.concatenate([self.default_root_pos, self.default_euler, self.default_q_actuated])
        # fmt: on
        # NOTE: there is some differences here in how retargeter state handles the initial
        # default state... particularly the root height as mentioned above
        self.default_retargeter_state = RetargeterState.initial()

        # STATE MACHINE STUFF
        self.state = State.STAND
        self.interpolate_start_time = None
        self.interpolation_duration = 2.0  # Seconds
        self.xrobo_t_prev = None

        # State initialization
        self.latest_xrobo_state = None

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
        self.xrobo_sub = self.create_subscription(
            XRoboState,
            "/xrobo/state",
            self.xr_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.teleop_pub = self.create_publisher(
            TeleopState,
            "/g1_control/teleop/state",
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.reset_pub = self.create_publisher(
            Empty, "/g1_control/reset", qos_reliable_keep_last_volatile_depth_1
        )
        self.estop_pub = self.create_publisher(
            Empty, "/g1_control/emergency_stop", qos_reliable_keep_last_volatile_depth_1
        )

        # fmt: off
        self.get_logger().info("====================================================")
        self.get_logger().info("Teleop node initialized")
        self.get_logger().info("Confirm that FULL BODY and CONTROLLER data is being sent.")
        self.get_logger().info("Please stand still with both feet on the ground.")
        self.get_logger().info("Then, press the A button to begin tracking.")
        self.get_logger().info("====================================================")
        # fmt: on

        self.timer = self.create_timer(self.dt_publish, self.teleop_step)
        self.get_logger().info("Teleop Node initialized")

    def xr_state_callback(self, msg: XRoboState) -> None:
        self.latest_xrobo_state = msg

    def get_current_time_in_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def reset(self):
        # Reset ROS node state management
        self.state = State.STAND
        self.interpolate_start_time = None
        self.xrobo_t_prev = None
        # Reset internal state management of the retargeter to a clean slate
        self.retargeter.state = self.default_retargeter_state

    def teleop_step(self):
        if self.latest_xrobo_state is None:
            self.get_logger().info("No XR data received yet", throttle_duration_sec=1.0)
            return

        if self.state == State.KILLED:
            self.get_logger().warn(
                "Robot is killed. Not sending data", throttle_duration_sec=1.0
            )
            return

        # Measure the elapsed time between the latest XRobo message and
        # the one used at the previous retargeting step
        cur_stamp = self.latest_xrobo_state.header.stamp
        t_cur = cur_stamp.sec + cur_stamp.nanosec * 1e-9
        if self.xrobo_t_prev is None:
            dt = self.dt_publish  # A reasonable default
        else:
            dt = t_cur - self.xrobo_t_prev
        self.xrobo_t_prev = t_cur

        # State machine handling
        # These buttons handle the transitions
        # A = Transition to TELEOP
        # X = Transition to STAND
        # Left Axis: Transition to KILLED
        # Right Axis: Reset simulation
        a_pressed = self.latest_xrobo_state.right_primary_button
        x_pressed = self.latest_xrobo_state.left_primary_button
        left_axis_clicked = self.latest_xrobo_state.left_axis_click
        right_axis_clicked = self.latest_xrobo_state.right_axis_click

        # Handle a kill command immediately
        if left_axis_clicked:
            self.get_logger().warn("Left axis clicked: Killing robot")
            self.estop_pub.publish(Empty())
            self.state = State.KILLED
            return

        # TODO also make sure this only happens if we are running in sim
        if right_axis_clicked:
            self.get_logger().info("Right axis clicked: Sending RESET command to sim")
            self.reset_pub.publish(Empty())
            self.reset()
            return

        ### RETARGETING LOGIC ###
        latest_teleop_data = np.reshape(
            self.latest_xrobo_state.flat_body_poses, (24, 7)
        )
        # The retargeter performs an initial SCP-based initialization step on the first call
        # In this case, we will compute the initialization and return without sending data
        # over ROS (just for this first call)
        if not self.retargeter._is_initialized:
            self.get_logger().warn(
                "Retargeter not initialized. Performing SCP initialization now"
            )
            self.retargeter.step(latest_teleop_data, dt)
            # Should be fully initialized at this point
            return
        # Call the retargeter (now fully initialized)
        q, pos, quat, vel, omega, contact_mode = self.retargeter.step(
            latest_teleop_data, dt
        )
        if np.any(np.isnan(q)):
            self.get_logger().error(f"NaNs detected. q={q}")
            self.estop_pub.publish(Empty())
            self.state = State.KILLED
            return

        # Handle state transition management
        current_interpolation_pct = 0.0
        if self.state == State.STAND:
            if a_pressed:
                # Add check to make sure that the current joint position is at least somewhat
                # close to the standing position... Sometimes, the data from XRobo can get into
                # a crazy state if it hasn't been calibrated in a while
                # Note that when making this check, we mainly want to look at the actuated DOFs
                threshold = 0.5  # Radians... Pretty loose threshold, but just avoids anything crazy
                q_close = np.all(np.abs(self.default_q_actuated - q[6:]) <= threshold)
                if q_close:
                    self.get_logger().info("Beginning interpolation to TELEOP")
                    self.state = State.INTERP_STAND_TO_TELEOP
                    self.interpolate_start_time = self.get_current_time_in_seconds()
                else:
                    self.get_logger().warn(
                        "WARNING: Detected a desired teleop pose that is far from the standing position."
                        + "\nNot transitioning to TELEOP mode until this is resolved, either by moving"
                        + "towards the standing position, or doing a recalibration"
                    )
        elif self.state == State.TELEOP:
            if x_pressed:
                self.get_logger().info("Beginning interpolation to STAND")
                self.state = State.INTERP_TELEOP_TO_STAND
                self.interpolate_start_time = self.get_current_time_in_seconds()
        elif self.state in [State.INTERP_STAND_TO_TELEOP, State.INTERP_TELEOP_TO_STAND]:
            current_time = self.get_current_time_in_seconds()
            current_interpolation_pct = (
                current_time - self.interpolate_start_time
            ) / self.interpolation_duration
            if current_interpolation_pct >= 1:
                if self.state == State.INTERP_STAND_TO_TELEOP:
                    self.get_logger().info(
                        "Done interpolation. Transitioning to TELEOP"
                    )
                    self.state = State.TELEOP
                    self.interpolate_start_time = None
                elif self.state == State.INTERP_TELEOP_TO_STAND:
                    self.get_logger().info("Done interpolation. Transitioning to STAND")
                    self.state = State.STAND
                    self.interpolate_start_time = None
                else:
                    raise RuntimeError("Unreachable state")
        else:
            # Note: should have returned already if KILLED
            raise RuntimeError("Unreachable state")

        # TODO: jit compile some of the interps together?
        msg = TeleopState()
        if self.state == State.STAND:
            msg.q = self.default_q_actuated.tolist()
            msg.position = self.default_root_pos.tolist()
            msg.quaternion = self.default_root_quat.tolist()
            msg.velocity = self.default_velocity.tolist()
            msg.omega = self.default_velocity.tolist()
            msg.contact_mode = int(self.default_contact_mode)
        elif self.state == State.INTERP_STAND_TO_TELEOP:
            q_interp = linear_interp(
                self.default_q_actuated, q[6:], current_interpolation_pct
            )
            pos_interp = linear_interp(
                self.default_root_pos, pos, current_interpolation_pct
            )
            quat_interp = slerp(self.default_root_quat, quat, current_interpolation_pct)
            vel_interp = linear_interp(
                self.default_velocity, vel, current_interpolation_pct
            )
            omega_interp = linear_interp(
                self.default_velocity, omega, current_interpolation_pct
            )
            msg.q = q_interp.tolist()
            msg.position = pos_interp.tolist()
            msg.quaternion = quat_interp.tolist()
            msg.velocity = vel_interp.tolist()
            msg.omega = omega_interp.tolist()
            msg.contact_mode = int(self.default_contact_mode)
        elif self.state == State.INTERP_TELEOP_TO_STAND:
            q_interp = linear_interp(
                q[6:], self.default_q_actuated, current_interpolation_pct
            )
            pos_interp = linear_interp(
                pos, self.default_root_pos, current_interpolation_pct
            )
            quat_interp = slerp(quat, self.default_root_quat, current_interpolation_pct)
            vel_interp = linear_interp(
                vel, self.default_velocity, current_interpolation_pct
            )
            omega_interp = linear_interp(
                omega, self.default_velocity, current_interpolation_pct
            )
            msg.q = q_interp.tolist()
            msg.position = pos_interp.tolist()
            msg.quaternion = quat_interp.tolist()
            msg.velocity = vel_interp.tolist()
            msg.omega = omega_interp.tolist()
            msg.contact_mode = int(self.default_contact_mode)
        elif self.state == State.TELEOP:
            msg.q = q[6:].tolist()
            msg.position = pos.tolist()
            msg.quaternion = quat.tolist()
            msg.velocity = vel.tolist()
            msg.omega = omega.tolist()
            msg.contact_mode = int(contact_mode)
        else:
            # Note: should have returned already if KILLED
            raise RuntimeError("Unreachable state")

        self.teleop_pub.publish(msg)


def linear_interp(start, end, pct):
    return start + pct * (end - start)


# HACK: copied jax version here and just changed it to numpy
def slerp(start: Array, end: Array, t: float) -> Array:
    """Spherical linear interpolation between two quaternions

    Note that the quaternion convention (WXYZ vs XYZW) does not matter,
    but the start and end must have the same convention

    Args:
        start (Array): Starting quaternion, shape (4,)
        end (Array): Ending quaternion, shape (4,)
        t (float): Interpolation parameter, 0 <= t <= 1.

    Returns:
        Array: Interpolated quaternion(s), shape (4,) or (n, 4)
    """
    # This parameter is used to resolve sign ambiguity in the quaternions
    # TODO: Make this an input that is marked as static
    shortest_path = True

    def _pick_closest_quaternion(q: Array, q_target: Array) -> Array:
        distance_flipped = np.linalg.norm(-q - q_target)
        distance_normal = np.linalg.norm(q - q_target)
        return np.where(distance_flipped < distance_normal, -q, q)

    def _angle_between_normalized_vectors(a: Array, b: Array) -> float:
        return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

    def _slerp_weights(angle: float, t: float) -> tuple:
        sin_angle = np.sin(angle)
        # Linear interpolation for very small angles
        w1_linear = 1.0 - t
        w2_linear = t
        # SLERP weights for larger angles
        w1_slerp = np.sin((1.0 - t) * angle) / sin_angle
        w2_slerp = np.sin(t * angle) / sin_angle
        # Decide which weights to use based on angle
        small_angle = angle < 1e-6
        w1 = np.where(small_angle, w1_linear, w1_slerp)
        w2 = np.where(small_angle, w2_linear, w2_slerp)
        return w1, w2

    assert start.shape == (4,)
    assert end.shape == (4,)
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    if shortest_path:
        end = _pick_closest_quaternion(end, start)
    angle = _angle_between_normalized_vectors(start, end)
    w1, w2 = _slerp_weights(angle, t)
    q = w1 * start + w2 * end
    # Renormalize to be safe
    return q / np.linalg.norm(q)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
