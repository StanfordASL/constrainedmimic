"""CBF configs for kinematic retargeting"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from cbfpy import CBFConfig

from frax import Humanoid


@jax.tree_util.register_static
class FullyActuatedConfig(CBFConfig):
    """CBF configuration for kinematic retargeting.

    These dynamics assume:
    - Fully actuated (including free-floating DOFs)
    - No contact constraints

    State: z = [q] (length = num_joints)
    Control: u = [joint velocities] (length = num_joints)

    Args:
        robot (Manipulator): The robot model (kinematics and dynamics)
        solver_tol (float, optional): Tolerance for the QP solver. Defaults to 1e-5.
        init_args (tuple, optional): Optional initial seed for testing additional CBFpy barrier arguments.
            Defaults to None.
        init_kwags (dict, optional): Optional initial seed for testing additional CBFpy barrier kwargs.
            Defaults to None.
    """

    def __init__(
        self,
        robot: Humanoid,
        solver_tol: float = 1e-5,
        init_args: Optional[tuple] = None,
        init_kwargs: Optional[dict] = None,
    ):
        assert isinstance(robot, Humanoid)
        assert isinstance(solver_tol, float) and 0.0 < solver_tol <= 1e-2
        assert init_args is None or isinstance(init_args, tuple)
        assert init_kwargs is None or isinstance(init_kwargs, dict)
        self.robot = robot
        self.num_joints = self.robot.num_joints
        super().__init__(
            n=self.num_joints,
            m=self.num_joints,
            u_min=-np.asarray(robot.joint_max_velocities),
            u_max=np.asarray(robot.joint_max_velocities),
            solver_tol=solver_tol,
            init_args=init_args,
            init_kwargs=init_kwargs,
            backend="elastiqp",
        )

    # Simple kinematic dynamics
    # - Assume fully actuated, including floating DOFs
    # - Assume no contact constraints

    def f(self, z, *args, **kwargs):
        return jnp.zeros(self.num_joints)

    def g(self, z, *args, **kwargs):
        return jnp.eye(self.num_joints)

    # NOTE: We will manually construct our objectives
    # in the retargeter, so these won't get used necessarily

    def P(self, z, u_des, *args, **kwargs):
        return super().P(z, u_des, *args, **kwargs)

    def q(self, z, u_des, *args, **kwargs):
        return super().q(z, u_des, *args, **kwargs)


@jax.tree_util.register_static
class JointLimitsFullyActuatedConfig(FullyActuatedConfig):
    """Example retargeting config with a simple joint limits CBF for testing purposes"""

    def __init__(
        self,
        robot: Humanoid,
        solver_tol: float = 1e-5,
        init_args: Optional[tuple] = None,
        init_kwargs: Optional[dict] = None,
    ):
        super().__init__(robot, solver_tol, init_args, init_kwargs)

    def alpha(self, h, *args, **kwargs):
        return 1.0 * h

    def h_1(self, z, *args, **kwargs):
        q = z
        q_min = jnp.asarray(self.robot.joint_lower_limits)
        q_max = jnp.asarray(self.robot.joint_upper_limits)
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])
        return h_joint_limits
