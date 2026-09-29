#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import argparse
import sys
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import argparse
import sys
from g1_control_msgs.msg import FrameState
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)


class EEStateVisualizer(Node):
    def __init__(self, side="left"):
        super().__init__(f"{side}_ee_state_visualizer")
        self.side = side

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Create a subscriber to the controller state topic
        topic = f"/xrobo/{side}_controller/state"
        self.subscription = self.create_subscription(
            FrameState,
            topic,
            self.ee_state_callback,
            qos_profile,
        )

        # Data storage for plotting - using fixed-length arrays
        self.buffer_size = 100
        self.time_data = np.linspace(-self.buffer_size + 1, 0, self.buffer_size)

        # Position data (x, y, z)
        self.pos_x = np.zeros(self.buffer_size)
        self.pos_y = np.zeros(self.buffer_size)
        self.pos_z = np.zeros(self.buffer_size)

        # Quaternion data (w, x, y, z)
        self.quat_w = np.zeros(self.buffer_size)
        self.quat_x = np.zeros(self.buffer_size)
        self.quat_y = np.zeros(self.buffer_size)
        self.quat_z = np.zeros(self.buffer_size)

        # Linear velocity data (x, y, z)
        self.vel_x = np.zeros(self.buffer_size)
        self.vel_y = np.zeros(self.buffer_size)
        self.vel_z = np.zeros(self.buffer_size)

        # Angular velocity data (x, y, z)
        self.ang_vel_x = np.zeros(self.buffer_size)
        self.ang_vel_y = np.zeros(self.buffer_size)
        self.ang_vel_z = np.zeros(self.buffer_size)

        # Track data ranges for auto-scaling
        self.min_max_values = {
            "pos_x": [float("inf"), float("-inf")],
            "pos_y": [float("inf"), float("-inf")],
            "pos_z": [float("inf"), float("-inf")],
            "quat_w": [float("inf"), float("-inf")],
            "quat_x": [float("inf"), float("-inf")],
            "quat_y": [float("inf"), float("-inf")],
            "quat_z": [float("inf"), float("-inf")],
            "vel_x": [float("inf"), float("-inf")],
            "vel_y": [float("inf"), float("-inf")],
            "vel_z": [float("inf"), float("-inf")],
            "ang_vel_x": [float("inf"), float("-inf")],
            "ang_vel_y": [float("inf"), float("-inf")],
            "ang_vel_z": [float("inf"), float("-inf")],
        }

        # Data lock to synchronize access from callback and animation
        self.data_lock = threading.Lock()

        # Flag to indicate if we've received data
        self.data_received = False

        # Setup the plot
        self.setup_plot()

        self.get_logger().info(
            f"EE State Visualizer for {side} side started on topic {topic}"
        )

    def update_min_max(self, name, value):
        """Update the min/max values for a data series"""
        if value < self.min_max_values[name][0]:
            self.min_max_values[name][0] = value
        if value > self.min_max_values[name][1]:
            self.min_max_values[name][1] = value

    def ee_state_callback(self, msg):
        try:
            with self.data_lock:
                # Update position data
                self.pos_x = np.roll(self.pos_x, -1)
                self.pos_y = np.roll(self.pos_y, -1)
                self.pos_z = np.roll(self.pos_z, -1)
                self.pos_x[-1] = msg.position[0]
                self.pos_y[-1] = msg.position[1]
                self.pos_z[-1] = msg.position[2]

                # Update quaternion data
                self.quat_w = np.roll(self.quat_w, -1)
                self.quat_x = np.roll(self.quat_x, -1)
                self.quat_y = np.roll(self.quat_y, -1)
                self.quat_z = np.roll(self.quat_z, -1)
                self.quat_w[-1] = msg.quaternion[0]
                self.quat_x[-1] = msg.quaternion[1]
                self.quat_y[-1] = msg.quaternion[2]
                self.quat_z[-1] = msg.quaternion[3]

                # Update linear velocity data
                self.vel_x = np.roll(self.vel_x, -1)
                self.vel_y = np.roll(self.vel_y, -1)
                self.vel_z = np.roll(self.vel_z, -1)
                self.vel_x[-1] = msg.velocity[0]
                self.vel_y[-1] = msg.velocity[1]
                self.vel_z[-1] = msg.velocity[2]

                # Update angular velocity data
                self.ang_vel_x = np.roll(self.ang_vel_x, -1)
                self.ang_vel_y = np.roll(self.ang_vel_y, -1)
                self.ang_vel_z = np.roll(self.ang_vel_z, -1)
                self.ang_vel_x[-1] = msg.omega[0]
                self.ang_vel_y[-1] = msg.omega[1]
                self.ang_vel_z[-1] = msg.omega[2]

                # Mark that we've received data
                self.data_received = True

            # Print some debug info occasionally (every ~100 callbacks)
            if np.random.random() < 0.01:
                self.get_logger().info(
                    f"Received {self.side} State: Pos X={msg.position[0]:.4f}, Vel X={msg.velocity[0]:.4f}"
                )
        except Exception as e:
            self.get_logger().error(f"Error in callback: {str(e)}")

    def setup_plot(self):
        # Create figure and subplots (4 rows, 4 columns)
        # Row 0: Position X, Y, Z
        # Row 1: Quaternion X, Y, Z, W
        # Row 2: Linear Velocity X, Y, Z
        # Row 3: Angular Velocity X, Y, Z
        self.fig, self.axs = plt.subplots(4, 4, figsize=(15, 12), dpi=100)
        plt.subplots_adjust(
            left=0.08, right=0.95, bottom=0.05, top=0.92, wspace=0.3, hspace=0.4
        )

        quat_min, quat_max = -1.1, 1.1
        pos_min, pos_max = -2, 2
        vel_min, vel_max = -1, 1
        ang_vel_min, ang_vel_max = -5, 5

        # Row 0: Position
        (self.pos_x_line,) = self.axs[0, 0].plot(
            self.time_data, self.pos_x, "r-", linewidth=2
        )
        self.axs[0, 0].set_title("Position X", fontsize=10, fontweight="bold")
        self.axs[0, 0].set_ylim(pos_min, pos_max)
        self.axs[0, 0].grid(True)

        (self.pos_y_line,) = self.axs[0, 1].plot(
            self.time_data, self.pos_y, "g-", linewidth=2
        )
        self.axs[0, 1].set_title("Position Y", fontsize=10, fontweight="bold")
        self.axs[0, 1].set_ylim(pos_min, pos_max)
        self.axs[0, 1].grid(True)

        (self.pos_z_line,) = self.axs[0, 2].plot(
            self.time_data, self.pos_z, "b-", linewidth=2
        )
        self.axs[0, 2].set_title("Position Z", fontsize=10, fontweight="bold")
        self.axs[0, 2].set_ylim(pos_min, pos_max)
        self.axs[0, 2].grid(True)
        self.axs[0, 3].set_visible(False)

        # Row 1: Quaternion
        (self.quat_w_line,) = self.axs[1, 0].plot(
            self.time_data, self.quat_w, "r-", linewidth=2
        )
        self.axs[1, 0].set_title("Quaternion W", fontsize=10, fontweight="bold")
        self.axs[1, 0].set_ylim(quat_min, quat_max)
        self.axs[1, 0].grid(True)

        (self.quat_x_line,) = self.axs[1, 1].plot(
            self.time_data, self.quat_x, "g-", linewidth=2
        )
        self.axs[1, 1].set_title("Quaternion X", fontsize=10, fontweight="bold")
        self.axs[1, 1].set_ylim(quat_min, quat_max)
        self.axs[1, 1].grid(True)

        (self.quat_y_line,) = self.axs[1, 2].plot(
            self.time_data, self.quat_y, "b-", linewidth=2
        )
        self.axs[1, 2].set_title("Quaternion Y", fontsize=10, fontweight="bold")
        self.axs[1, 2].set_ylim(quat_min, quat_max)
        self.axs[1, 2].grid(True)

        (self.quat_z_line,) = self.axs[1, 3].plot(
            self.time_data, self.quat_z, "m-", linewidth=2
        )
        self.axs[1, 3].set_title("Quaternion Z", fontsize=10, fontweight="bold")
        self.axs[1, 3].set_ylim(quat_min, quat_max)
        self.axs[1, 3].grid(True)

        # Row 2: Linear Velocity
        (self.vel_x_line,) = self.axs[2, 0].plot(
            self.time_data, self.vel_x, "r--", linewidth=1.5
        )
        self.axs[2, 0].set_title("Vel X", fontsize=10, fontweight="bold")
        self.axs[2, 0].set_ylim(vel_min, vel_max)
        self.axs[2, 0].grid(True)

        (self.vel_y_line,) = self.axs[2, 1].plot(
            self.time_data, self.vel_y, "g--", linewidth=1.5
        )
        self.axs[2, 1].set_title("Vel Y", fontsize=10, fontweight="bold")
        self.axs[2, 1].set_ylim(vel_min, vel_max)
        self.axs[2, 1].grid(True)

        (self.vel_z_line,) = self.axs[2, 2].plot(
            self.time_data, self.vel_z, "b--", linewidth=1.5
        )
        self.axs[2, 2].set_title("Vel Z", fontsize=10, fontweight="bold")
        self.axs[2, 2].set_ylim(vel_min, vel_max)
        self.axs[2, 2].grid(True)
        self.axs[2, 3].set_visible(False)

        # Row 3: Angular Velocity
        (self.ang_vel_x_line,) = self.axs[3, 0].plot(
            self.time_data, self.ang_vel_x, "r:", linewidth=1.5
        )
        self.axs[3, 0].set_title("Omega X", fontsize=10, fontweight="bold")
        self.axs[3, 0].set_ylim(ang_vel_min, ang_vel_max)
        self.axs[3, 0].grid(True)

        (self.ang_vel_y_line,) = self.axs[3, 1].plot(
            self.time_data, self.ang_vel_y, "g:", linewidth=1.5
        )
        self.axs[3, 1].set_title("Omega Y", fontsize=10, fontweight="bold")
        self.axs[3, 1].set_ylim(ang_vel_min, ang_vel_max)
        self.axs[3, 1].grid(True)

        (self.ang_vel_z_line,) = self.axs[3, 2].plot(
            self.time_data, self.ang_vel_z, "b:", linewidth=1.5
        )
        self.axs[3, 2].set_title("Omega Z", fontsize=10, fontweight="bold")
        self.axs[3, 2].set_ylim(ang_vel_min, ang_vel_max)
        self.axs[3, 2].grid(True)
        self.axs[3, 3].set_visible(False)

        self.fig.suptitle(
            f"XRobo {self.side.capitalize()} Controller State Visualization",
            fontsize=16,
            fontweight="bold",
        )

    def update_plot(self, frame):
        with self.data_lock:
            # Make local copies
            pos_x, pos_y, pos_z = (
                self.pos_x.copy(),
                self.pos_y.copy(),
                self.pos_z.copy(),
            )
            quat_w, quat_x, quat_y, quat_z = (
                self.quat_w.copy(),
                self.quat_x.copy(),
                self.quat_y.copy(),
                self.quat_z.copy(),
            )
            vel_x, vel_y, vel_z = (
                self.vel_x.copy(),
                self.vel_y.copy(),
                self.vel_z.copy(),
            )
            ang_vel_x, ang_vel_y, ang_vel_z = (
                self.ang_vel_x.copy(),
                self.ang_vel_y.copy(),
                self.ang_vel_z.copy(),
            )

        self.pos_x_line.set_ydata(pos_x)
        self.pos_y_line.set_ydata(pos_y)
        self.pos_z_line.set_ydata(pos_z)

        self.quat_w_line.set_ydata(quat_w)
        self.quat_x_line.set_ydata(quat_x)
        self.quat_y_line.set_ydata(quat_y)
        self.quat_z_line.set_ydata(quat_z)

        self.vel_x_line.set_ydata(vel_x)
        self.vel_y_line.set_ydata(vel_y)
        self.vel_z_line.set_ydata(vel_z)

        self.ang_vel_x_line.set_ydata(ang_vel_x)
        self.ang_vel_y_line.set_ydata(ang_vel_y)
        self.ang_vel_z_line.set_ydata(ang_vel_z)

        return [
            self.pos_x_line,
            self.pos_y_line,
            self.pos_z_line,
            self.quat_w_line,
            self.quat_x_line,
            self.quat_y_line,
            self.quat_z_line,
            self.vel_x_line,
            self.vel_y_line,
            self.vel_z_line,
            self.ang_vel_x_line,
            self.ang_vel_y_line,
            self.ang_vel_z_line,
        ]

    def start_animation(self):
        # Create animation
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=None,
            interval=50,
            blit=True,
            cache_frame_data=False,
        )
        plt.show()


class ROS2Thread(threading.Thread):
    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node
        self.daemon = True

    def run(self):
        rclpy.spin(self.node)


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize XRobo controller pose data")
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Controller side to visualize (left or right)",
    )

    # Filter out ROS2 specific arguments before passing to argparse
    ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)
    params, _ = parser.parse_known_args(ros_args[1:])

    rclpy.init(args=args)

    # Create the node
    visualizer = EEStateVisualizer(side=params.side)

    # Start ROS2 spin in separate thread
    ros_thread = ROS2Thread(visualizer)
    ros_thread.start()

    try:
        # Run matplotlib animation in the main thread
        visualizer.start_animation()
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
