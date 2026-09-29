import numpy as np

from cm_control.config.g1_config import floating_root_g1_joint_ordering

PICO_IDX_TO_NAME = {
    0: "Pelvis",
    1: "Left_Hip",
    2: "Right_Hip",
    3: "Spine1",
    4: "Left_Knee",
    5: "Right_Knee",
    6: "Spine2",
    7: "Left_Ankle",
    8: "Right_Ankle",
    9: "Spine3",
    10: "Left_Foot",
    11: "Right_Foot",
    12: "Neck",
    13: "Left_Collar",
    14: "Right_Collar",
    15: "Head",
    16: "Left_Shoulder",
    17: "Right_Shoulder",
    18: "Left_Elbow",
    19: "Right_Elbow",
    20: "Left_Wrist",
    21: "Right_Wrist",
    22: "Left_Hand",
    23: "Right_Hand",
}
PICO_NAME_TO_IDX = {n: i for i, n in PICO_IDX_TO_NAME.items()}

PICO_RETARGETING_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Left_Knee",
    "Left_Foot",
    "Right_Hip",
    "Right_Knee",
    "Right_Foot",
    "Spine3",
    "Left_Shoulder",
    "Left_Elbow",
    "Left_Wrist",
    "Right_Shoulder",
    "Right_Elbow",
    "Right_Wrist",
)
PICO_RETARGETING_IDXS = tuple(PICO_NAME_TO_IDX[n] for n in PICO_RETARGETING_NAMES)

G1_IDX_TO_NAME = {i: n for i, n in enumerate(floating_root_g1_joint_ordering)}
G1_NAME_TO_IDX = {n: i for i, n in G1_IDX_TO_NAME.items()}
G1_RETARGETING_NAMES = (
    "yaw_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
)
G1_RETARGETING_IDXS = tuple(G1_NAME_TO_IDX[n] for n in G1_RETARGETING_NAMES)

PICO_TO_G1_SCALING_FACTORS = (
    0.76,  # pelvis
    0.9,  # left_hip_yaw_link
    0.95,  # left_knee_link
    0.76,  # left_toe_link
    0.9,  # right_hip_yaw_link
    0.95,  # right_knee_link
    0.76,  # right_toe_link
    0.9,  # torso_link
    0.8,  # left_shoulder_yaw_link
    0.85,  # left_elbow_link
    0.85,  # left_wrist_yaw_link
    0.8,  # right_shoulder_yaw_link
    0.85,  # right_elbow_link
    0.85,  # right_wrist_yaw_link
)
# np.array(
#     [
#         [0.9, 0.9, 0.76],  # pelvis
#         [0.9, 0.9, 0.9],  # left_hip_yaw_link
#         [0.95, 0.95, 0.95],  # left_knee_link
#         [0.9, 0.9, 0.76],  # left_toe_link
#         [0.9, 0.9, 0.9],  # right_hip_yaw_link
#         [0.95, 0.95, 0.95],  # right_knee_link
#         [0.9, 0.9, 0.76],  # right_toe_link
#         [0.9, 0.9, 0.9],  # torso_link
#         [0.8, 0.8, 0.8],  # left_shoulder_yaw_link
#         [0.85, 0.85, 0.85],  # left_elbow_link
#         [0.85, 0.85, 0.85],  # left_wrist_yaw_link
#         [0.8, 0.8, 0.8],  # right_shoulder_yaw_link
#         [0.85, 0.85, 0.85],  # right_elbow_link
#         [0.85, 0.85, 0.85],  # right_wrist_yaw_link
#     ]
# )

# TODO decide if these tuning values should be here or somewhere else
RETARGET_KP_POS = 10 * np.ones(14)
RETARGET_KP_ROT = 5 * np.ones(14)
RETARGET_WEIGHT_POS = np.array(
    [
        10.0,  # pelvis
        10.0,  # left_hip_yaw_link
        10.0,  # left_knee_link
        100.0,  # left_toe_link
        10.0,  # right_hip_yaw_link
        10.0,  # right_knee_link
        100.0,  # right_toe_link
        0.0,  # torso_link
        0.0,  # left_shoulder_yaw_link
        0.0,  # left_elbow_link
        1.0,  # left_wrist_yaw_link
        0.0,  # right_shoulder_yaw_link
        0.0,  # right_elbow_link
        1.0,  # right_wrist_yaw_link
    ]
)
RETARGET_WEIGHT_ROT = np.array(
    [
        5.0,  # pelvis
        5.0,  # left_hip_yaw_link
        5.0,  # left_knee_link
        10.0,  # left_toe_link
        5.0,  # right_hip_yaw_link
        5.0,  # right_knee_link
        10.0,  # right_toe_link
        10.0,  # torso_link
        10.0,  # left_shoulder_yaw_link
        10.0,  # left_elbow_link
        10.0,  # left_wrist_yaw_link
        10.0,  # right_shoulder_yaw_link
        10.0,  # right_elbow_link
        10.0,  # right_wrist_yaw_link
    ]
)
