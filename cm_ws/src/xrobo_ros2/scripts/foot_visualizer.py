#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import sys
from xrobo_ros2.msg import XRoboState
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)


class FootVisualizer(Node):
    def __init__(self):
        super().__init__("foot_visualizer")

        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Buffer size for history
        self.buffer_size = 100
        self.time_data = np.linspace(-self.buffer_size + 1, 0, self.buffer_size)

        # Data structures for Right Foot (Index 11)
        self.foot_data = {
            "pos": np.zeros((self.buffer_size, 3)),
            "quat": np.zeros((self.buffer_size, 4)),  # w, x, y, z
            "vel": np.zeros((self.buffer_size, 3)),
            "omega": np.zeros((self.buffer_size, 3)),
        }

        self.sub = self.create_subscription(
            XRoboState, "/xrobo/state", self.callback, qos_profile
        )

        self.data_lock = threading.Lock()
        self.setup_plot()

    def callback(self, msg):
        # Index 11 for right foot
        idx = 11
        pos_start = idx * 7

        if len(msg.flat_body_poses) < (pos_start + 7):
            return

        with self.data_lock:
            # Update Position
            self.foot_data["pos"] = np.roll(self.foot_data["pos"], -1, axis=0)
            self.foot_data["pos"][-1] = msg.flat_body_poses[pos_start : pos_start + 3]

            # Update Orientation (SDK/ROS convention in XRoboState: w, x, y, z)
            self.foot_data["quat"] = np.roll(self.foot_data["quat"], -1, axis=0)
            self.foot_data["quat"][-1] = msg.flat_body_poses[
                pos_start + 3 : pos_start + 7
            ]

            # Update Velocity
            self.foot_data["vel"] = np.roll(self.foot_data["vel"], -1, axis=0)
            self.foot_data["vel"][-1] = msg.right_foot_twist[:3]

            # Update Angular Velocity
            self.foot_data["omega"] = np.roll(self.foot_data["omega"], -1, axis=0)
            self.foot_data["omega"][-1] = msg.right_foot_twist[3:]

    def setup_plot(self):
        # 4 rows, 4 columns
        # Row 0: Pos X, Y, Z
        # Row 1: Ori Qw, Qx, Qy, Qz
        # Row 2: Vel X, Y, Z
        # Row 3: Omega X, Y, Z
        self.fig, self.axs = plt.subplots(4, 4, figsize=(16, 12))
        plt.subplots_adjust(
            hspace=0.4, wspace=0.3, left=0.05, right=0.98, top=0.9, bottom=0.05
        )

        titles = [
            ["Pos X", "Pos Y", "Pos Z", ""],
            ["Ori Qw", "Ori Qx", "Ori Qy", "Ori Qz"],
            ["Vel X", "Vel Y", "Vel Z", ""],
            ["Omega X", "Omega Y", "Omega Z", ""],
        ]

        self.lines = {}
        for r in range(4):
            for c in range(4):
                ax = self.axs[r, c]
                title = titles[r][c]
                if not title:
                    ax.set_visible(False)
                    continue

                ax.set_title(title, fontsize=10)
                ax.set_ylim(-1.5, 1.5)
                if r == 0:  # Pos
                    ax.set_ylim(-3, 3)
                if r >= 2:  # Velocity/Omega
                    ax.set_ylim(-10, 10)
                ax.grid(True, alpha=0.3)

                label = f"row{r}_col{c}"
                color = ["r", "g", "b", "m"][r]
                (l,) = ax.plot(self.time_data, np.zeros(self.buffer_size), color=color)
                self.lines[label] = l

        self.fig.suptitle(
            "Right Foot State Visualization (Index 11)", fontsize=16, fontweight="bold"
        )

    def update_plot(self, frame):
        with self.data_lock:
            # Row 0: Pos
            for i in range(3):
                self.lines[f"row0_col{i}"].set_ydata(self.foot_data["pos"][:, i])

            # Row 1: Ori [Qw, Qx, Qy, Qz]
            for i in range(4):
                self.lines[f"row1_col{i}"].set_ydata(self.foot_data["quat"][:, i])

            # Row 2: Vel
            for i in range(3):
                self.lines[f"row2_col{i}"].set_ydata(self.foot_data["vel"][:, i])

            # Row 3: Omega
            for i in range(3):
                self.lines[f"row3_col{i}"].set_ydata(self.foot_data["omega"][:, i])

        return list(self.lines.values())

    def start(self):
        ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        plt.show()


def main():
    rclpy.init()
    node = FootVisualizer()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.start()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
