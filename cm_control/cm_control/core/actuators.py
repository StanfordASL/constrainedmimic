"""Actuator Models"""

# TODO: Check to see if there is any difference in actuator configuation between beyondmimic and twist2

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from numpy.typing import ArrayLike

from cm_control.config.g1_config import fixed_root_g1_joint_ordering


@jax.tree_util.register_static
class RobotActuatorModel:
    """PD actuator model

    Args:
        stiffness (ArrayLike): Joint stiffnesses, shape (num_joints,)
        damping (ArrayLike): Joint damping, shape (num_joints,)
        max_effort (ArrayLike): Maximum joint efforts, shape (num_joints,)
    """

    def __init__(self, stiffness: ArrayLike, damping: ArrayLike, max_effort: ArrayLike):
        self.stiffness = np.asarray(stiffness).ravel()
        self.damping = np.asarray(damping).ravel()
        self.max_effort = np.asarray(max_effort).ravel()
        self.num_joints = self.stiffness.shape[0]
        assert self.damping.shape[0] == self.num_joints
        assert self.max_effort.shape[0] == self.num_joints
        assert np.all(self.stiffness > 0)
        assert np.all(self.damping > 0)
        assert np.all(self.max_effort > 0)

    def compute_joint_torques(
        self,
        q_cur: Array,
        qdot_cur: Array,
        q_des: Array,
        qdot_des: Array,
        tau_des: Array,
    ) -> Array:
        """Computes joint torques based on a simple PD actuator model

        Args:
            q_cur (Array): Current joint positions, shape (num_joints,)
            qdot_cur (Array): Current joint velocities, shape (num_joints,)
            q_des (Array): Desired joint positions, shape (num_joints,)
            qdot_des (Array): Desired joint velocities, shape (num_joints,)
            tau_des (Array): Desired feed-forward torque, shape (num_joints,)

        Returns:
            Array: Joint torques, shape (num_joints,)
        """
        assert q_cur.shape == (self.num_joints,)
        assert qdot_cur.shape == (self.num_joints,)
        assert q_des.shape == (self.num_joints,)
        assert qdot_des.shape == (self.num_joints,)
        assert tau_des.shape == (self.num_joints,)

        # Note: this implementation matches how IsaacLab handles an ImplicitActuator
        # which was used in beyondmimic
        # Errors
        error_pos = q_des - q_cur
        error_vel = qdot_des - qdot_cur
        # Desired (unclipped) joint torques
        computed_effort = (
            self.stiffness * error_pos + self.damping * error_vel + tau_des
        )
        # Clip the torques based on the motor limits
        applied_effort = jnp.clip(computed_effort, -self.max_effort, self.max_effort)
        return applied_effort

    def compute_joint_torques_from_only_positional_target(
        self, q_cur: Array, qdot_cur: Array, q_des: Array
    ) -> Array:
        """Compute joint torques, assuming no desired joint velocities or feedforward torque is provided

        Args:
            q_cur (Array): Current joint positions, shape (num_joints,)
            qdot_cur (Array): Current joint velocities, shape (num_joints,)
            q_des (Array): Desired joint positions, shape (num_joints,)

        Returns:
            Array: Joint torques, shape (num_joints,)
        """
        # If the policy outputs just a joint position target, we can simplify things
        # and set the ff torque and desired joint velocity to (static) zero vectors
        tau_des = jnp.zeros(self.num_joints)
        qdot_des = jnp.zeros(self.num_joints)
        return self.compute_joint_torques(q_cur, qdot_cur, q_des, qdot_des, tau_des)

    def get_positional_target_from_torques(
        self, q_cur: Array, qdot_cur: Array, tau_des: Array
    ) -> Array:
        """Compute the positional target to achieve a desired joint torque from a PD actuator

        Args:
            q_cur (Array): Current joint positions, shape (num_joints,)
            qdot_cur (Array): Current joint velocities, shape (num_joints,)
            tau_des (Array): Desired joint torques, shape (num_joints,)

        Returns:
            Array: Joint positions, shape (num_joints,)
        """
        assert q_cur.shape == (self.num_joints,)
        assert qdot_cur.shape == (self.num_joints,)
        assert tau_des.shape == (self.num_joints,)
        # Inverse mapping. Does not make any assumptions about if tau has been clipped to the effort limits
        q_des = q_cur + (1 / self.stiffness) * (tau_des + self.damping * qdot_cur)
        # Note: not clipping to positional limits
        return q_des


@jax.tree_util.register_static
class G1ActuatorModel(RobotActuatorModel):
    """PD actuator model for the Unitree G1, based on values from BeyondMimic"""

    # Actuator values are from BeyondMimic
    ARMATURE_5020 = 0.003609725
    ARMATURE_7520_14 = 0.010177520
    ARMATURE_7520_22 = 0.025101925
    ARMATURE_4010 = 0.00425

    NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
    DAMPING_RATIO = 2.0

    STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
    STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
    STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
    STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

    DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
    DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
    DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
    DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

    def __init__(self):
        stiffnesses = []
        dampings = []
        max_efforts = []
        for joint_name in fixed_root_g1_joint_ordering:
            # Legs
            if joint_name.endswith(("hip_yaw_joint", "hip_pitch_joint")):
                stiffnesses.append(self.STIFFNESS_7520_14)
                dampings.append(self.DAMPING_7520_14)
                max_efforts.append(88.0)
            elif joint_name.endswith(("hip_roll_joint", "knee_joint")):
                stiffnesses.append(self.STIFFNESS_7520_22)
                dampings.append(self.DAMPING_7520_22)
                max_efforts.append(139.0)
            # Feet
            elif joint_name.endswith(("ankle_pitch_joint", "ankle_roll_joint")):
                stiffnesses.append(2.0 * self.STIFFNESS_5020)
                dampings.append(2.0 * self.DAMPING_5020)
                max_efforts.append(50.0)
            # Waist
            elif joint_name in ("waist_roll_joint", "waist_pitch_joint"):
                stiffnesses.append(2.0 * self.STIFFNESS_5020)
                dampings.append(2.0 * self.DAMPING_5020)
                max_efforts.append(50.0)
            elif joint_name == "waist_yaw_joint":
                stiffnesses.append(self.STIFFNESS_7520_14)
                dampings.append(self.DAMPING_7520_14)
                max_efforts.append(88.0)
            # Arms
            elif joint_name.endswith(
                (
                    "shoulder_pitch_joint",
                    "shoulder_roll_joint",
                    "shoulder_yaw_joint",
                    "elbow_joint",
                    "wrist_roll_joint",
                ),
            ):
                stiffnesses.append(self.STIFFNESS_5020)
                dampings.append(self.DAMPING_5020)
                max_efforts.append(25.0)
            elif joint_name.endswith(("wrist_pitch_joint", "wrist_yaw_joint")):
                stiffnesses.append(self.STIFFNESS_4010)
                dampings.append(self.DAMPING_4010)
                max_efforts.append(5.0)
            else:
                raise ValueError("Unrecognized joint name: ", joint_name)
        super().__init__(stiffnesses, dampings, max_efforts)
