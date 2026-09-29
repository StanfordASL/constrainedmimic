import jax
import jax.numpy as jnp
import numpy as np


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = jnp.stack([x, y, z, w], axis=-1).reshape(shape)

    return quat


def normalize(x, eps: float = 1e-9):
    return x / jnp.clip(jnp.linalg.norm(x, axis=-1, keepdims=True), min=eps)


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = jnp.cross(xyz, b) * 2
    return (b + a[:, 3:] * t + jnp.cross(xyz, t)).reshape(shape)


def quat_rotate(q, v):
    shape = v.shape
    q_w = q[..., -1:]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = jnp.cross(q_vec, v) * q_w * 2.0
    c = q_vec * jnp.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    shape = v.shape
    q_w = q[..., -1:]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = jnp.cross(q_vec, v) * q_w * 2.0
    c = q_vec * jnp.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return jnp.concatenate([-a[:, :3], a[:, -1:]], axis=-1).reshape(shape)


def quat_unit(a):
    return normalize(a)


def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).reshape(*angle.shape, 1)
    xyz = normalize(axis) * jnp.sin(theta)
    w = jnp.cos(theta)
    return quat_unit(jnp.concatenate([xyz, w], axis=-1))


def normalize_angle(x):
    return jnp.atan2(jnp.sin(x), jnp.cos(x))


def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


def tf_apply(q, t, v):
    return quat_apply(q, v) + t


def tf_vector(q, v):
    return quat_apply(q, v)


def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


def get_basis_vector(q, v):
    return quat_rotate(q, v)


def copysign(a, b):
    return jnp.abs(a) * jnp.sign(b)


def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = (
        q[..., qw] * q[..., qw]
        - q[..., qx] * q[..., qx]
        - q[..., qy] * q[..., qy]
        + q[..., qz] * q[..., qz]
    )
    roll = jnp.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = jnp.where(jnp.abs(sinp) >= 1, copysign(jnp.pi / 2.0, sinp), jnp.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = (
        q[..., qw] * q[..., qw]
        + q[..., qx] * q[..., qx]
        - q[..., qy] * q[..., qy]
        - q[..., qz] * q[..., qz]
    )
    yaw = jnp.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
    # TODO: This differs from the torch and makes a big difference in the policy performance
    # Figure out why this is the case
    # return roll % (2 * jnp.pi), pitch % (2 * jnp.pi), yaw % (2 * jnp.pi)


def quat_from_euler_xyz(roll, pitch, yaw):
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return jnp.stack([qx, qy, qz, qw], axis=-1)


def tensor_clamp(t, min_t, max_t):
    return jnp.maximum(jnp.minimum(t, max_t), min_t)


def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def euler_from_quaternion(quat_angle):
    x = quat_angle[..., 0]
    y = quat_angle[..., 1]
    z = quat_angle[..., 2]
    w = quat_angle[..., 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = jnp.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = jnp.clip(t2, -1, 1)
    pitch_y = jnp.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = jnp.atan2(t3, t4)

    return jnp.stack([roll_x, pitch_y, yaw_z], axis=-1)


def quat_to_angle_axis(q):
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = jnp.sqrt(jnp.clip(1 - q[..., qw] * q[..., qw], min=0.0))
    angle = 2 * jnp.acos(jnp.clip(q[..., qw], min=-1.0, max=1.0))
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta[..., jnp.newaxis]
    axis = q[..., qx:qw] / jnp.clip(sin_theta_expand, min=min_theta)

    mask = jnp.abs(sin_theta) > min_theta
    default_axis = jnp.zeros_like(axis)
    default_axis = default_axis.at[..., -1].set(1.0)

    angle = jnp.where(mask, angle, jnp.zeros_like(angle))
    axis = jnp.where(mask[..., jnp.newaxis], axis, default_axis)
    return angle, axis


def angle_axis_to_exp_map(angle, axis):
    return angle[..., jnp.newaxis] * axis


def quat_to_exp_map(q):
    angle, axis = quat_to_angle_axis(q)
    return angle_axis_to_exp_map(angle, axis)


def quat_to_tan_norm(q):
    ref_tan = jnp.zeros_like(q[..., 0:3])
    ref_tan = ref_tan.at[..., 0].set(1.0)
    tan = quat_rotate(q, ref_tan)

    ref_norm = jnp.zeros_like(q[..., 0:3])
    ref_norm = ref_norm.at[..., -1].set(1.0)
    norm = quat_rotate(q, ref_norm)

    return jnp.concatenate([tan, norm], axis=-1)


def euler_xyz_to_exp_map(roll, pitch, yaw):
    q = quat_from_euler_xyz(roll, pitch, yaw)
    return quat_to_exp_map(q)


def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = jnp.linalg.norm(exp_map, axis=-1)
    axis = exp_map / jnp.clip(angle[..., jnp.newaxis], min=min_theta)
    angle = normalize_angle(angle)

    default_axis = jnp.zeros_like(exp_map)
    default_axis = default_axis.at[..., -1].set(1.0)

    mask = jnp.abs(angle) > min_theta
    angle = jnp.where(mask, angle, jnp.zeros_like(angle))
    axis = jnp.where(mask[..., jnp.newaxis], axis, default_axis)

    return angle, axis


def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    return quat_from_angle_axis(angle, axis)


def slerp(q0, q1, t):
    cos_half_theta = jnp.sum(q0 * q1, axis=-1)

    neg_mask = cos_half_theta < 0
    q1 = jnp.where(neg_mask[..., jnp.newaxis], -q1, q1)

    cos_half_theta = jnp.abs(cos_half_theta)

    half_theta = jnp.acos(jnp.clip(cos_half_theta, min=-1.0, max=1.0))
    sin_half_theta = jnp.sqrt(jnp.clip(1.0 - cos_half_theta * cos_half_theta, min=0.0))

    ratioA = jnp.sin((1 - t) * half_theta) / jnp.clip(sin_half_theta, min=1e-6)
    ratioB = jnp.sin(t * half_theta) / jnp.clip(sin_half_theta, min=1e-6)

    new_q = ratioA[..., jnp.newaxis] * q0 + ratioB[..., jnp.newaxis] * q1

    new_q = jnp.where(
        jnp.abs(sin_half_theta)[..., jnp.newaxis] < 0.001, 0.5 * q0 + 0.5 * q1, new_q
    )
    new_q = jnp.where(cos_half_theta[..., jnp.newaxis] >= 1, q0, new_q)

    return new_q


def calc_heading(q):
    ref_dir = jnp.zeros_like(q[..., 0:3])
    ref_dir = ref_dir.at[..., 0].set(1.0)
    rot_dir = quat_rotate(q, ref_dir)

    heading = jnp.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


def calc_heading_quat(q):
    heading = calc_heading(q)
    axis = jnp.zeros_like(q[..., 0:3])
    axis = axis.at[..., 2].set(1.0)
    return quat_from_angle_axis(heading, axis)


def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = jnp.zeros_like(q[..., 0:3])
    axis = axis.at[..., 2].set(1.0)
    return quat_from_angle_axis(-heading, axis)


def quat_pos(x):
    z = x[..., 3:] < 0
    return (1 - 2 * z) * x


def quat_to_axis_angle(q):
    eps = 1e-5
    q = quat_pos(q)
    length = jnp.linalg.norm(q[..., 0:3], axis=-1)

    angle = 2.0 * jnp.atan2(length, q[..., 3])
    axis = q[..., 0:3] / jnp.clip(length[..., jnp.newaxis], min=eps)

    default_axis = jnp.zeros_like(axis)
    default_axis = default_axis.at[..., -1].set(1.0)
    mask = length > eps

    angle = jnp.where(mask, angle, jnp.zeros_like(angle))
    axis = jnp.where(mask[..., jnp.newaxis], axis, default_axis)

    return axis, angle


def quat_diff(q0, q1):
    return quat_mul(q1, quat_conjugate(q0))


def quat_diff_angle(q0, q1):
    dq = quat_diff(q0, q1)
    _, angle = quat_to_axis_angle(dq)
    return angle
