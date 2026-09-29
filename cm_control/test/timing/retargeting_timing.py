"""Retargeting timing test"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from frax import load_g1
from cbfpy import CBF

from cm_control.core.kinematic_configs import BaseKinematicConfig
from cm_control.retargeting.pico_to_g1_retarget import (
    PicoToG1Retargeter,
    RetargeterState,
)
from timing_utils import benchmark_function


class TimingKinematicCBFConfig(BaseKinematicConfig):
    def __init__(self, robot, num_cbfs_to_test):
        # This is super hacky but will work well for this timing-only test
        # We want to see: As we increase the number of CBFs, how does the
        # compute frequency decrease?
        # We'll evaluate this in steps of 50
        assert num_cbfs_to_test % 50 == 0 and num_cbfs_to_test <= 300
        # self.num_cbfs_to_test = num_cbfs_to_test

        # If we add an obstacle into the environment and assign a CBF for
        # collision avoidance for all spheres on the robot body, this will
        # generate 46 constraints (for 46 spheres). We can "fill in" the
        # remaining 4 constraints for each multiple of 50 with some of the
        # joint limit CBFs
        # So, let's spawn some obstacles to test
        num_robot_spheres = 46
        self.num_obstacles = num_cbfs_to_test // num_robot_spheres
        self.num_joint_limit_cbfs = self.num_obstacles * (50 - num_robot_spheres)
        np.random.seed(0)
        # Sample some points approximately on a cylinder around the robot
        distance = np.random.uniform(0.75, 1.5, size=(self.num_obstacles,))
        thetas = np.random.uniform(0, 2 * np.pi, size=(self.num_obstacles,))
        z = np.random.uniform(-0.5, 0.5, size=(self.num_obstacles,))
        x = distance * np.cos(thetas)
        y = distance * np.sin(thetas)
        self.obstacle_positions = np.column_stack([x, y, z])
        self.obstacle_radii = np.random.uniform(0.1, 0.3, size=(self.num_obstacles,))

        super().__init__(
            constrained=True,
            underactuated=False,
            robot=robot,
            # NOTE: Since this CBF will get integrated into the retargeting, the
            # objective function will change. So the use_naive_objective flag
            # doesn't really do anything for this
            use_naive_objective=True,
            solver_tol=1e-5,
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_1(self, z, *args, **kwargs):
        q = z

        robot_collision_positions, robot_collision_radii = (
            self.robot.link_collision_data(q)
        )
        center_deltas = (
            robot_collision_positions[:, None, :] - self.obstacle_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.obstacle_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=-1) - radii_sums

        h_joint_lower_limits = q - jnp.asarray(self.robot.joint_lower_limits)
        h_joint_upper_limits = jnp.asarray(self.robot.joint_upper_limits) - q
        h_joint_limits = jnp.concatenate([h_joint_lower_limits, h_joint_upper_limits])
        return jnp.concatenate(
            [h_collision, h_joint_limits[: self.num_joint_limit_cbfs]]
        )

    def alpha(self, h, *args, **kwargs):
        return 10.0 * h


def run(num_cbfs: int):

    robot = load_g1()

    if num_cbfs <= 0:
        kin_cbf = None
    else:
        kin_cbf_config = TimingKinematicCBFConfig(robot, num_cbfs)
        kin_cbf = CBF.from_config(kin_cbf_config)

    inner_retargeter = PicoToG1Retargeter(kin_cbf, robot)

    # Initialize state
    state = RetargeterState.initial()

    dummy_poses = np.zeros((24, 7))
    dummy_dt = 0.01
    args = (dummy_poses, state, dummy_dt)

    avg_time, jit_time = benchmark_function(inner_retargeter.step, args, n_calls=10000)
    avg_time_ms = avg_time * 1e3
    avg_hz = 1 / avg_time
    print("JIT time (seconds): ", jit_time)
    print("Average time (milliseconds): ", avg_time_ms)
    print("Average frequency (Hz): ", avg_hz)
    return avg_time, jit_time


def run_all():
    num_cbfs = [0, 50, 100, 150, 200, 250, 300]
    avg_times = []
    jit_times = []
    for num in num_cbfs:
        print("=" * 20)
        print(f"[{Path(__file__).name}]: Starting timing experiment.")
        print(f"[{Path(__file__).name}]: Number of CBFs: {num}")
        avg_time, jit_time = run(num)
        avg_times.append(avg_time)
        jit_times.append(jit_time)
        print("=" * 20)

    # Save results to CSV
    output_path = Path(__file__).parent / "retargeting_timing_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(["num_cbfs", "avg_time", "jit_time"])
        # Data rows
        for num, avg, jit in zip(num_cbfs, avg_times, jit_times):
            writer.writerow([num, avg, jit])


if __name__ == "__main__":
    run_all()
