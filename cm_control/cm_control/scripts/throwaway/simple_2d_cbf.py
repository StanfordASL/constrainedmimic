import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
from cbfpy import CBF, CBFConfig


XML = """
<mujoco model="xy_puck_arena">
    <compiler assetdir="assets" />
    <option timestep="0.01" gravity="0 0 0" />

    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" />

        <geom name="floor" type="plane" size="5 5 .01" rgba="0.8 0.8 0.8 1" 
              contype="0" conaffinity="0" />

        <!-- Robot -->
        <body name="robot" pos="0 0 0">
            <joint name="robot_x" type="slide" axis="1 0 0" />
            <joint name="robot_y" type="slide" axis="0 1 0" />
            <geom name="robot_geom" type="cylinder" size="0.1 0.05" rgba="0 0.7 0 1" mass="1" />
        </body>

        <!-- Obstacle -->
        <body name="obstacle" pos="0 0 0">
            <joint name="obstacle_x" type="slide" axis="1 0 0" />
            <joint name="obstacle_y" type="slide" axis="0 1 0" />
            <geom name="obstacle_geom" type="cylinder" size="0.1 0.05" rgba="0.7 0 0 1" mass="1" />
        </body>
    </worldbody>

    <!-- Velocity control for the robot -->
    <actuator>
        <velocity name="v_control_x" joint="robot_x" kv="10" />
        <velocity name="v_control_y" joint="robot_y" kv="10" />
    </actuator>
</mujoco>
"""


@jax.tree_util.register_static
class DynamicObstacle2DCBFConfig(CBFConfig):
    def __init__(self):
        self.radius = 0.1
        self.obstacle_radius = 0.1
        self.lookahead_time = 2.0
        self.padding = 0.1
        max_vel = 1.0  # Tune this
        super().__init__(
            n=4,  # XY position and velocity
            m=2,  # XY velocity
            u_min=-max_vel * np.ones(2),
            u_max=max_vel * np.ones(2),
            init_kwargs={"z_obs": np.ones(4)},  # Initial seed for obstacle state
            backend="elastiqp",
        )

    def f(self, z, *args, **kwargs):
        # Assume direct control over the robot velocity
        return jnp.zeros(self.n)

    def g(self, z, *args, **kwargs):
        return jnp.vstack([jnp.eye(2), jnp.zeros((2, 2))])
        # return super().g(z, *args, **kwargs)

    def h_1(self, z, *args, **kwargs):
        z_obs = kwargs["z_obs"]
        pos_obs = z_obs[:2]
        vel_obs = z_obs[2:]
        pos = z[:2]
        vel = z[2:]
        distance = jnp.linalg.norm(pos - pos_obs)
        dir_obs_to_robot = (pos - pos_obs) / distance
        collision_velocity_component = (vel_obs - vel).T @ dir_obs_to_robot
        return jnp.array(
            [
                distance
                - collision_velocity_component * self.lookahead_time
                - self.obstacle_radius
                - self.radius
                - self.padding
            ]
        )

    def alpha(self, h, *args, **kwargs):
        return 5.0 * h


def nominal_controller(z, z_des):
    Kp = 1.0
    Kd = 1.0
    return Kp * (z_des[:2] - z[:2]) + Kd * (z_des[2:] - z[2:])


def main():
    cbf_config = DynamicObstacle2DCBFConfig()
    cbf = CBF.from_config(cbf_config)

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[:] = np.array([0, 0, 1, 1])
    mujoco.mj_forward(model, data)

    # We want the robot to be at the origin with zero velocity
    z_des = np.zeros(4)  # static

    dt = 1 / 100

    @jax.jit
    def safe_controller(z, z_obs):
        u_nom = nominal_controller(z, z_des)
        return cbf.safety_filter(z, u_nom, z_obs=z_obs)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            z = np.concatenate([data.qpos[:2], data.qvel[:2]])
            z_obs = np.concatenate([data.qpos[2:], data.qvel[2:]])
            u = safe_controller(z, z_obs)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
