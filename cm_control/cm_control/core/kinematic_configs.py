"""Kinematic CBF configs"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from cbfpy import CBFConfig

from frax import Humanoid
from cm_control.core.control_affine import (
    f_kin_simple,
    g_kin_simple,
    f_kin_constrained,
    g_kin_constrained,
    f_kin_constrained_underactuated,
    g_kin_constrained_underactuated,
)
from cm_control.core.kinematic_objectives import P_qp, q_qp_from_P


@jax.tree_util.register_static
class BaseKinematicConfig(CBFConfig):
    """Base CBFConfig for kinematic (velocity) control"""

    def __init__(
        self,
        constrained: bool,
        underactuated: bool,
        robot: Humanoid,
        use_naive_objective: bool = False,
        solver_tol: float = 1e-5,
        init_args: Optional[tuple] = None,
        init_kwargs: Optional[dict] = None,
    ):
        assert isinstance(constrained, bool)
        assert isinstance(underactuated, bool)
        assert not (not constrained and underactuated)  # not implemented
        assert isinstance(robot, Humanoid)
        assert isinstance(use_naive_objective, bool)
        assert isinstance(solver_tol, float) and 0.0 < solver_tol <= 1e-2
        assert init_args is None or isinstance(init_args, tuple)
        assert init_kwargs is None or isinstance(init_kwargs, dict)
        if constrained:
            assert "contact_mode" in init_kwargs
        self.constrained = constrained
        self.underactuated = underactuated
        self.robot = robot
        self.use_naive_objective = use_naive_objective

        n = self.robot.num_joints
        u_min = -np.asarray(robot.joint_max_velocities)
        u_max = np.asarray(robot.joint_max_velocities)
        if underactuated:
            u_min = u_min[-robot.num_actuated_joints :]
            u_max = u_max[-robot.num_actuated_joints :]
            m = self.robot.num_actuated_joints
        else:
            m = self.robot.num_joints

        super().__init__(
            n=n,
            m=m,
            u_min=u_min,
            u_max=u_max,
            solver_tol=solver_tol,
            init_args=init_args,
            init_kwargs=init_kwargs,
            backend="elastiqp",
        )

    def f(self, z, *args, **kwargs):
        if self.constrained:
            contact_mode = kwargs["contact_mode"]
            if self.underactuated:
                return f_kin_constrained_underactuated(self.robot, z, contact_mode)
            else:  # Fully actuated
                return f_kin_constrained(self.robot, z, contact_mode)
        else:  # Not constrained
            if self.underactuated:
                raise NotImplementedError()
            else:  # Fully actuated
                return f_kin_simple(self.robot, z)

    def g(self, z, *args, **kwargs):
        if self.constrained:
            contact_mode = kwargs["contact_mode"]
            if self.underactuated:
                return g_kin_constrained_underactuated(self.robot, z, contact_mode)
            else:  # Fully actuated
                return g_kin_constrained(self.robot, z, contact_mode)
        else:  # Not constrained
            if self.underactuated:
                raise NotImplementedError()
            else:  # Fully actuated
                return g_kin_simple(self.robot, z)

    def P(self, z, u_des, *args, **kwargs):
        if self.use_naive_objective:
            return super().P(z, u_des, *args, **kwargs)
        # Constrained and fully actuated is all that I've implemented so far
        assert self.constrained and not self.underactuated
        # TODO: MOVE THESE
        w_foot_motion = jnp.ones(6)
        w_hand = jnp.ones(6)
        w_com = jnp.ones(3)
        w_null = jnp.ones(self.robot.num_joints)

        q = z[: self.robot.num_joints]
        contact_mode = kwargs["contact_mode"]
        return P_qp(
            self.robot,
            q,
            contact_mode,
            w_foot_motion,
            w_hand,
            w_com,
            w_null,
        )

    def q(self, z, u_des, *args, **kwargs):
        if self.use_naive_objective:
            return super().q(z, u_des, *args, **kwargs)
        # Constrained and fully actuated is all that I've implemented so far
        assert self.constrained and not self.underactuated
        P = self.P(z, u_des, *args, **kwargs)
        return q_qp_from_P(P, u_des)

    # In inherited classes, still need to define:
    # h_1/h_2, alpha_1/alpha_2
