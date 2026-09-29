"""Unitree G1 configuration info

Note: most values can be determined from the URDF in the assets directory.
These config values are specific to how I'm handling the G1
(custom joint ordering, motor values, ...)
"""

import numpy as np


fixed_root_g1_joint_ordering = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
fixed_root_g1_link_ordering = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

floating_root_g1_joint_ordering = [
    "x_prismatic",
    "y_prismatic",
    "z_prismatic",
    "roll_joint",
    "pitch_joint",
    "yaw_joint",
] + fixed_root_g1_joint_ordering

floating_root_g1_link_ordering = [
    "world",
    "x_link",
    "y_link",
    "z_link",
    "roll_link",
    "pitch_link",
] + fixed_root_g1_link_ordering

fixed_root_g1_link_to_parent_joint_mapping = {
    "left_hip_pitch_link": "left_hip_pitch_joint",
    "left_hip_roll_link": "left_hip_roll_joint",
    "left_hip_yaw_link": "left_hip_yaw_joint",
    "left_knee_link": "left_knee_joint",
    "left_ankle_pitch_link": "left_ankle_pitch_joint",
    "left_ankle_roll_link": "left_ankle_roll_joint",
    "right_hip_pitch_link": "right_hip_pitch_joint",
    "right_hip_roll_link": "right_hip_roll_joint",
    "right_hip_yaw_link": "right_hip_yaw_joint",
    "right_knee_link": "right_knee_joint",
    "right_ankle_pitch_link": "right_ankle_pitch_joint",
    "right_ankle_roll_link": "right_ankle_roll_joint",
    "waist_yaw_link": "waist_yaw_joint",
    "waist_roll_link": "waist_roll_joint",
    "torso_link": "waist_pitch_joint",
    "left_shoulder_pitch_link": "left_shoulder_pitch_joint",
    "left_shoulder_roll_link": "left_shoulder_roll_joint",
    "left_shoulder_yaw_link": "left_shoulder_yaw_joint",
    "left_elbow_link": "left_elbow_joint",
    "left_wrist_roll_link": "left_wrist_roll_joint",
    "left_wrist_pitch_link": "left_wrist_pitch_joint",
    "left_wrist_yaw_link": "left_wrist_yaw_joint",
    "right_shoulder_pitch_link": "right_shoulder_pitch_joint",
    "right_shoulder_roll_link": "right_shoulder_roll_joint",
    "right_shoulder_yaw_link": "right_shoulder_yaw_joint",
    "right_elbow_link": "right_elbow_joint",
    "right_wrist_roll_link": "right_wrist_roll_joint",
    "right_wrist_pitch_link": "right_wrist_pitch_joint",
    "right_wrist_yaw_link": "right_wrist_yaw_joint",
}

floating_root_g1_link_to_parent_joint_mapping = {
    "x_link": "x_prismatic",
    "y_link": "y_prismatic",
    "z_link": "z_prismatic",
    "roll_link": "roll_joint",
    "pitch_link": "pitch_joint",
    "pelvis": "yaw_joint",
} | fixed_root_g1_link_to_parent_joint_mapping

fixed_root_g1_joint_to_child_link_mapping = {
    j: l for l, j in fixed_root_g1_link_to_parent_joint_mapping.items()
}
floating_root_g1_joint_to_child_link_mapping = {
    j: l for l, j in floating_root_g1_link_to_parent_joint_mapping.items()
}
dummy_link_names = {"world", "x_link", "y_link", "z_link", "roll_link", "pitch_link"}


ankle_indices = np.array([4, 5, 10, 11])
"""Indices of ankle joints in the 29DOF G1, assuming the joint ordering used by me/mujoco/pinocchio"""

# Note that things like the joint position/velocity/torque limits can also be parsed from the URDF
# But, there are some places where we need this data outside of the Humanoid class
# fmt: off
joint_max_torques = np.array([
    88.0, 139.0, 88.0, 139.0, 35.0, 35.0,  # Left leg
    88.0, 139.0, 88.0, 139.0, 35.0, 35.0,  # Right leg
    88.0, 35.0, 35.0,  # Waist
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,  # Left arm
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,  # Right arm
])
"""Maximum joint torques"""
# fmt: on

foot_com_to_ground_distance = 0.013575
"""How far above the ground is the COM of the foot, when standing?"""
