#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import argparse
import sys
from xrobo_ros2.msg import XRoboState
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)


class FrameVisualizer(Node):
    def __init__(self, frame_name="right_controller"):
        super().__init__("frame_visualizer")

        self.frame_name = frame_name
        self.get_logger().info(f"Visualizing frame: {self.frame_name}")

        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Buffer size for history
        self.buffer_size = 100
        self.time_data = np.linspace(-self.buffer_size + 1, 0, self.buffer_size)

        # Data structure
        self.data = {
            "pos": np.zeros((self.buffer_size, 3)),
            "quat": np.zeros((self.buffer_size, 4)),  # w, x, y, z
            "twist_lin": np.zeros((self.buffer_size, 3)),
            "twist_ang": np.zeros((self.buffer_size, 3)),
        }

        self.sub = self.create_subscription(
            XRoboState, "/xrobo/state", self.callback, qos_profile
        )

        self.data_lock = threading.Lock()
        self.setup_plot()

    def callback(self, msg):
        found = False
        pos = [0.0, 0.0, 0.0]
        quat = [1.0, 0.0, 0.0, 0.0]
        twist_lin = [0.0, 0.0, 0.0]
        twist_ang = [0.0, 0.0, 0.0]

        if self.frame_name == "headset":
            pos = msg.headset_pose[:3]
            quat = msg.headset_pose[3:]
            twist_lin = msg.headset_twist[:3]
            twist_ang = msg.headset_twist[3:]
            found = True
        elif self.frame_name == "left_controller":
            pos = msg.left_controller_pose[:3]
            quat = msg.left_controller_pose[3:]
            twist_lin = msg.left_controller_twist[:3]
            twist_ang = msg.left_controller_twist[3:]
            found = True
        elif self.frame_name == "right_controller":
            pos = msg.right_controller_pose[:3]
            quat = msg.right_controller_pose[3:]
            twist_lin = msg.right_controller_twist[:3]
            twist_ang = msg.right_controller_twist[3:]
            found = True
        elif self.frame_name == "left_foot":
            # Index 10
            idx = 10
            pos = msg.flat_body_poses[idx * 7 : idx * 7 + 3]
            quat = msg.flat_body_poses[idx * 7 + 3 : idx * 7 + 7]
            twist_lin = msg.left_foot_twist[:3]
            twist_ang = msg.left_foot_twist[3:]
            found = True
        elif self.frame_name == "right_foot":
            # Index 11
            idx = 11
            pos = msg.flat_body_poses[idx * 7 : idx * 7 + 3]
            quat = msg.flat_body_poses[idx * 7 + 3 : idx * 7 + 7]
            twist_lin = msg.right_foot_twist[:3]
            twist_ang = msg.right_foot_twist[3:]
            found = True
        elif self.frame_name == "root" or self.frame_name == "pelvis":
            # Index 0
            idx = 0
            pos = msg.flat_body_poses[idx * 7 : idx * 7 + 3]
            quat = msg.flat_body_poses[idx * 7 + 3 : idx * 7 + 7]
            twist_lin = msg.root_twist[:3]
            twist_ang = msg.root_twist[3:]
            found = True

        if found:
            with self.data_lock:
                self.data["pos"] = np.roll(self.data["pos"], -1, axis=0)
                self.data["pos"][-1] = pos
                self.data["quat"] = np.roll(self.data["quat"], -1, axis=0)
                self.data["quat"][-1] = quat
                self.data["twist_lin"] = np.roll(self.data["twist_lin"], -1, axis=0)
                self.data["twist_lin"][-1] = twist_lin
                self.data["twist_ang"] = np.roll(self.data["twist_ang"], -1, axis=0)
                self.data["twist_ang"][-1] = twist_ang

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
            ["Lin Vel X", "Lin Vel Y", "Lin Vel Z", ""],
            ["Ang Vel X", "Ang Vel Y", "Ang Vel Z", ""],
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
            f"XRobo Frame Visualization: {self.frame_name.upper()}",
            fontsize=16,
            fontweight="bold",
        )

    def update_plot(self, frame):
        with self.data_lock:
            # Row 0: Pos
            for i in range(3):
                self.lines[f"row0_col{i}"].set_ydata(self.data["pos"][:, i])

            # Row 1: Ori [Qw, Qx, Qy, Qz]
            for i in range(4):
                self.lines[f"row1_col{i}"].set_ydata(self.data["quat"][:, i])

            # Row 2: Vel
            for i in range(3):
                self.lines[f"row2_col{i}"].set_ydata(self.data["twist_lin"][:, i])

            # Row 3: Omega
            for i in range(3):
                self.lines[f"row3_col{i}"].set_ydata(self.data["twist_ang"][:, i])

        return list(self.lines.values())

    def start(self):
        ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize XRobo frames")
    parser.add_argument(
        "--frame",
        type=str,
        default="right_controller",
        choices=[
            "headset",
            "left_controller",
            "right_controller",
            "left_foot",
            "right_foot",
            "root",
            "pelvis",
        ],
        help="Frame to visualize",
    )
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = FrameVisualizer(frame_name=args.frame)

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
