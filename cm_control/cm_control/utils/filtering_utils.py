from typing import Optional, Tuple

from jax import Array

from cm_control.utils.rotation_utils import slerp, omega_world_from_wxyz_quaternions


def position_filter(
    pos: Array,
    vel: Optional[Array],
    last_pos: Array,
    last_vel: Array,
    alpha_pos: Optional[float],
    alpha_vel: Optional[float],
    dt: float,
) -> Tuple[Array, Array]:
    """Apply an exponential moving average (EMA) filter to position data

    Args:
        pos (Array): Current position, shape (3,)
        vel (Optional[Array]): Current velocity, shape (3,). If None, will infer
            the current velocity from the change in position.
        last_pos (Array): Last position, shape (3,)
        last_vel (Array): Last velocity, shape (3,)
        alpha_pos (Optional[float]): EMA filter parameter for position, 0 < alpha <= 1.
            None if not applying smoothing.
        alpha_vel (Optional[float]): EMA filter parameter for velocity, 0 < alpha <= 1
            None if not applying smoothing.
        dt (float): Timestep

    Returns:
        Tuple[Array, Array]:
            filtered_pos (Array): Filtered position, shape (3,)
            filtered_vel (Array): Filtered velocity, shape (3,)
    """
    if alpha_pos is not None:
        filtered_pos = alpha_pos * pos + (1.0 - alpha_pos) * last_pos
    else:
        filtered_pos = pos

    if vel is None:
        vel = (pos - last_pos) / dt

    if alpha_vel is not None:
        filtered_vel = alpha_vel * vel + (1.0 - alpha_vel) * last_vel
    else:
        filtered_vel = vel

    return filtered_pos, filtered_vel


def orientation_filter(
    quat: Array,
    omega: Optional[Array],
    last_quat: Array,
    last_omega: Array,
    alpha_quat: Optional[float],
    alpha_omega: Optional[float],
    dt: float,
) -> Tuple[Array, Array]:
    """Apply an exponential moving average (EMA) filter to orientation data

    Args:
        quat (Array): Current WXYZ quaternion, shape (4,)
        omega (Optional[Array]): Current (WORLD-FRAME) angular velocity, shape (3,). If None,
            will infer the current angular velocity from the change in quaternion.
        last_quat (Array): Last WXYZ quaternion, shape (4,)
        last_omega (Array): Last (WORLD-FRAME) angular velocity, shape (3,)
        alpha_quat (Optional[float]): SLERP filter parameter for quaternion, 0 < alpha <= 1
            None if not applying smoothing.
        alpha_omega (Optional[float]): EMA filter parameter for angular velocity, 0 < alpha <= 1
            None if not applying smoothing.
        dt (float): Timestep

    Returns:
        Tuple[Array, Array]:
            filtered_quat (Array): Filtered WXYZ quaternion, shape (3,)
            filtered_omega (Array): Filtered (WORLD-FRAME) angular velocity, shape (3,)
    """
    if alpha_quat is not None:
        filtered_quat = slerp(last_quat, quat, alpha_quat)
    else:
        filtered_quat = quat

    if omega is None:
        omega = omega_world_from_wxyz_quaternions(last_quat, quat, dt)

    if alpha_omega is not None:
        filtered_omega = alpha_omega * omega + (1.0 - alpha_omega) * last_omega
    else:
        filtered_omega = omega

    return filtered_quat, filtered_omega


def pose_filter(
    pos: Array,
    quat: Array,
    vel: Optional[Array],
    omega: Optional[Array],
    last_pos: Array,
    last_quat: Array,
    last_vel: Array,
    last_omega: Array,
    alpha_pos: Optional[float],
    alpha_quat: Optional[float],
    alpha_vel: Optional[float],
    alpha_omega: Optional[float],
    dt: float,
) -> Tuple[Array, Array, Array, Array]:
    """Apply an exponential moving average (EMA) filter to pose data

    Args:
        pos (Array): Current position, shape (3,)
        quat (Array): Current WXYZ quaternion, shape (4,)
        vel (Optional[Array]): Current velocity, shape (3,). If None, will infer
            the current velocity from the change in position.
        omega (Optional[Array]): Current (WORLD-FRAME) angular velocity, shape (3,). If None,
            will infer the current angular velocity from the change in quaternion.
        last_pos (Array): Last position, shape (3,)
        last_quat (Array): Last WXYZ quaternion, shape (4,)
        last_vel (Array): Last velocity, shape (3,)
        last_omega (Array): Last (WORLD-FRAME) angular velocity, shape (3,)
        alpha_pos (Optional[float]): EMA filter parameter for position, 0 < alpha <= 1.
            None if not applying smoothing.
        alpha_quat (Optional[float]): SLERP filter parameter for quaternion, 0 < alpha <= 1
            None if not applying smoothing.
        alpha_vel (Optional[float]): EMA filter parameter for velocity, 0 < alpha <= 1
            None if not applying smoothing.
        alpha_omega (Optional[float]): EMA filter parameter for angular velocity, 0 < alpha <= 1
            None if not applying smoothing.
        dt (float): Timestep

    Returns:
        Tuple[Array, Array, Array, Array]:
            filtered_pos (Array): Filtered position, shape (3,)
            filtered_quat (Array): Filtered WXYZ quaternion, shape (3,)
            filtered_vel (Array): Filtered velocity, shape (3,)
            filtered_omega (Array): Filtered (WORLD-FRAME) angular velocity, shape (3,)
    """
    filtered_pos, filtered_vel = position_filter(
        pos, vel, last_pos, last_vel, alpha_pos, alpha_vel, dt
    )
    filtered_quat, filtered_omega = orientation_filter(
        quat, omega, last_quat, last_omega, alpha_quat, alpha_omega, dt
    )
    return filtered_pos, filtered_quat, filtered_vel, filtered_omega
