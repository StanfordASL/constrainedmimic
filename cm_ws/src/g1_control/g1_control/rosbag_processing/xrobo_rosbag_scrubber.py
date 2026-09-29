"""Interactive viewer for XRobo data.

Allows for manual movement through the bag data and viewing
of the time indices associated with the data
"""

import time
import argparse

import mujoco
import mujoco.viewer

from g1_control.rosbag_processing.rosbag_loading import load_ros2_bag


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


class OfflineBodyVisualizer:
    def __init__(self, messages):
        self.messages = messages
        self.idx = 0
        self.playing = False
        self.start_idx = None
        self.end_idx = None

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
        self.body_name_to_idx = {n: i for i, n in enumerate(self.body_names)}

        self.num_bodies = 27  # 1 + 24 + 2
        self.model = mujoco.MjModel.from_xml_string(create_mjcf(self.num_bodies))
        self.data = mujoco.MjData(self.model)

        self.step_size = 1
        self.coarse_step = 20
        self.fine_step = 1

        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )

        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -15
        self.viewer.cam.lookat = [0, 0, -1.0]

    def apply_message(self, msg):
        self.data.mocap_pos[0] = msg["headset_pose"][:3]
        self.data.mocap_quat[0] = msg["headset_pose"][3:]

        for i in range(24):
            idx = 1 + i
            base = i * 7

            self.data.mocap_pos[idx] = msg["flat_body_poses"][base : base + 3]
            quat = msg["flat_body_poses"][base + 3 : base + 7]

            self.data.mocap_quat[idx] = quat

        self.data.mocap_pos[25] = msg["left_controller_pose"][:3]
        self.data.mocap_quat[25] = msg["left_controller_pose"][3:]

        self.data.mocap_pos[26] = msg["right_controller_pose"][:3]
        self.data.mocap_quat[26] = msg["right_controller_pose"][3:]

    def key_callback(self, key):
        if key == 262:  # →
            self.idx = self.idx = min(self.idx + self.step_size, len(self.messages) - 1)
            self.print_status()

        elif key == 263:  # ←
            self.idx = self.idx = max(self.idx - self.step_size, 0)
            self.print_status()

        elif key == 32:  # space
            self.playing = not self.playing

        elif key == ord("["):
            self.start_idx = self.idx
            print(f"[START] {self.start_idx}")

        elif key == ord("]"):
            self.end_idx = self.idx
            print(f"[END] {self.end_idx}")

        elif key == ord("/"):
            if self.step_size == self.fine_step:
                self.step_size = self.coarse_step
                print(f"[MODE] Coarse ({self.step_size} frames)")
            else:
                self.step_size = self.fine_step
                print(f"[MODE] Fine ({self.step_size} frame)")

    def print_status(self):
        t0 = self.messages[0]["timestamp"]
        t = (self.messages[self.idx]["timestamp"] - t0) * 1e-9
        print(f"idx={self.idx}, t={t:.3f}s")

    def run(self):
        while self.viewer.is_running():
            if self.playing:
                self.idx = min(self.idx + 1, len(self.messages) - 1)

            self.apply_message(self.messages[self.idx])

            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()

            time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    args = parser.parse_args()

    histories = load_ros2_bag(args.bag)
    messages = histories["/xrobo/state"]

    print(f"Loaded {len(messages)} frames")

    vis = OfflineBodyVisualizer(messages)
    vis.run()

    # After closing viewer → report selection
    print("\n==== Selection ====")
    print(f"start_idx: {vis.start_idx}")
    print(f"end_idx:   {vis.end_idx}")

    if vis.start_idx is not None and vis.end_idx is not None:
        t0 = messages[0]["timestamp"]
        t_start = (messages[vis.start_idx]["timestamp"] - t0) * 1e-9
        t_end = (messages[vis.end_idx]["timestamp"] - t0) * 1e-9

        print(f"t_start: {t_start:.3f}s")
        print(f"t_end:   {t_end:.3f}s")


if __name__ == "__main__":
    main()
