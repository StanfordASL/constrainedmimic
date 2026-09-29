"""Script to step through all of the spheres in the G1's collision model and indicate their indices"""

import mujoco
import mujoco.viewer
import numpy as np
from frax import load_g1

from cm_control.assets import G1_XML
from cm_control.utils.mujoco_utils import visualize_sphere
from cm_control.utils.rotation_utils import quat_wxyz_to_intrinsic_euler_xyz


def main():
    xml_path = str(G1_XML)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    robot = load_g1()
    q = np.concatenate(
        [data.qpos[:3], quat_wxyz_to_intrinsic_euler_xyz(data.qpos[3:7]), data.qpos[7:]]
    )
    collision_pos, collision_radii = robot.link_collision_data(q)
    n_spheres = collision_pos.shape[0]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_spheres):
            if not viewer.is_running():
                break
            pos = collision_pos[i]
            rad = collision_radii[i]
            visualize_sphere(viewer, pos, rad, (1, 0, 0, 0.5))
            viewer.sync()
            print("Index: ", i)
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
