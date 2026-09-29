import time
from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer

from frax import load_g1, Humanoid
from cm_control.assets import G1_XML
from cm_control.utils.mujoco_utils import visualize_transform, visualize_sphere
from cm_control.utils.assorted_utils import mujoco_qpos_to_q
from cm_control.core.support_polygon import (
    left_foot_support_points,
    right_foot_support_points,
)


def main():
    xml_path = str(G1_XML)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    robot = load_g1()

    np.random.seed(0)
    random_pos = np.random.rand(3)
    random_quat = np.random.rand(4)
    random_quat /= np.linalg.norm(random_quat)
    random_q_act = np.random.rand(29)
    qpos_random = np.concatenate([random_pos, random_quat, random_q_act])
    data.qpos[:] = qpos_random

    q_mine = mujoco_qpos_to_q(data.qpos)
    transform = robot.left_foot_transform(q_mine)
    points = left_foot_support_points(robot, q_mine)
    right_foot_transform = robot.right_foot_transform(q_mine)
    right_foot_points = right_foot_support_points(robot, q_mine)
    num_points_per_foot = 4

    with mujoco.viewer.launch_passive(model, data) as viewer:
        visualize_transform(viewer, transform)
        visualize_transform(viewer, right_foot_transform)
        for i in range(num_points_per_foot):
            visualize_sphere(viewer, points[i], size=0.01, rgba=(1, 0, 0, 0.25))
            visualize_sphere(
                viewer, right_foot_points[i], size=0.01, rgba=(0, 1, 0, 0.25)
            )
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
