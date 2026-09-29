"""Config values for anything related to the TWIST2 controller"""

import numpy as np

ang_vel_scale = 0.25
dof_vel_scale = 0.05
dof_pos_scale = 1.0
action_scale = 0.5
# fmt: off
default_joint_position = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
    0.0, 0.0, 0.0, # torso (3)
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
])
default_mimic_obs = np.array([
    0.0, 0.0, # X/Y velocity in body frame
    0.8, # Root z height
    0.0, 0.0, # Roll and pitch
    0.0, # Yaw angular velocity in body frame
    *default_joint_position, # 29-DOF joint angles
])
# Joint PD controller parameters
# TODO: do some tuning. TWIST2 has slightly different values
# for sim and real... But only for the arm joints?
hardware_kps = np.array([
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,  # Left leg
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,  # Right leg
    150.0, 150.0, 150.0,  # Waist
    40.0, 40.0, 40.0, 40.0, 20.0, 20.0, 20.0,  # Left arm
    40.0, 40.0, 40.0, 40.0, 20.0, 20.0, 20.0,  # Right arm
])
hardware_kds =  np.array([
      2.0, 2.0, 2.0, 4.0, 2.0, 2.0,  # Left leg
      2.0, 2.0, 2.0, 4.0, 2.0, 2.0,  # Right leg
      4.0, 4.0, 4.0,  # Waist
      5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0,  # Left arm
      5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0,  # Right arm
])
sim_kps = np.array([
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,  # Left leg
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,  # Right leg
    150.0, 150.0, 150.0,  # Waist
    40.0, 40.0, 40.0, 40.0, 4.0, 4.0, 4.0,  # Left arm
    40.0, 40.0, 40.0, 40.0, 4.0, 4.0, 4.0,  # Right arm
])
sim_kds =  np.array([
      2.0, 2.0, 2.0, 4.0, 2.0, 2.0,  # Left leg
      2.0, 2.0, 2.0, 4.0, 2.0, 2.0,  # Right leg
      4.0, 4.0, 4.0,  # Waist
      5.0, 5.0, 5.0, 5.0, 0.2, 0.2, 0.2,  # Left arm
      5.0, 5.0, 5.0, 5.0, 0.2, 0.2, 0.2,  # Right arm
])
