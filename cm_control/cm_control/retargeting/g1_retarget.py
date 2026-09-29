"""Retargeting for the Unitree G1"""

import time

import jax.numpy as jnp
import numpy as np
from cbfpy import CBF

from frax import load_g1
from cm_control.retargeting.retarget import KinematicRetargeter
from cm_control.retargeting.retargeting_cbf_configs import (
    JointLimitsFullyActuatedConfig,
)
from cm_control.config.retargeting_config import G1_RETARGETING_IDXS


def main():
    robot = load_g1()
    # Using joint limits as an example
    cbf_config = JointLimitsFullyActuatedConfig(robot)
    cbf = CBF.from_config(cbf_config)
    idxs = jnp.asarray(G1_RETARGETING_IDXS)

    # TODO tune these values
    kp_pos = 1.0
    kp_rot = 1.0
    weight_pos = 1.0
    weight_rot = 0.1
    stepsize = 1.0
    convergence_tol = 1e-3
    max_iters = 30
    retargeter = KinematicRetargeter(
        cbf,
        robot,
        idxs,
        kp_pos,
        kp_rot,
        weight_pos,
        weight_rot,
        stepsize,
        convergence_tol,
        max_iters,
    )

    # Test it out via retargeting from g1 to g1
    # (Simple test case where we know exactly what the result should be)
    np.random.seed(0)
    q_init = np.zeros(robot.num_joints)
    q_perturbed = q_init + np.random.uniform(-0.1, 0.1, robot.num_joints)
    # Use the perturbed joint positions as the reference for the retargeting
    all_link_tfs = robot.link_to_world_transforms(q_perturbed)
    all_link_positions = all_link_tfs[:, :3, 3]
    all_link_rotations = all_link_tfs[:, :3, :3]
    # Use only the links of interest for retargeting
    selected_link_positions = all_link_positions[idxs]
    selected_link_rotations = all_link_rotations[idxs]
    # Retarget to these values
    q_out, converged, iters = retargeter.scp_retarget(
        q_init, selected_link_positions, selected_link_rotations
    )

    # Compute task-space errors
    all_link_tfs_out = robot.link_to_world_transforms(q_out)
    selected_link_positions_out = all_link_tfs_out[idxs, :3, 3]
    pos_err = np.linalg.norm(
        selected_link_positions_out - selected_link_positions, axis=-1
    )

    print("Mean Joint Error: ", np.mean(np.abs(q_out - q_perturbed)))
    print("Max Joint Error: ", np.max(np.abs(q_out - q_perturbed)))
    print("Mean Task Pos Error: ", np.mean(pos_err))
    print("Max Task Pos Error: ", np.max(pos_err))
    print("Converged?: ", converged)
    print("Number of iterations: ", iters)

    # Timing test
    # Same logic as before, but now with timing
    # We assume that the first call from earlier completed the JIT
    num_tests = 10
    all_times = []
    all_iters = []
    for _ in range(num_tests):
        q_perturbed = q_init + np.random.uniform(-0.1, 0.1, robot.num_joints)
        all_link_tfs = robot.link_to_world_transforms(q_perturbed)
        all_link_positions = all_link_tfs[:, :3, 3]
        all_link_rotations = all_link_tfs[:, :3, :3]
        # Use only the links of interest for retargeting
        selected_link_positions = all_link_positions[idxs]
        selected_link_rotations = all_link_rotations[idxs]
        start_time = time.perf_counter()
        # Retarget to these values
        q_out, converged, iters = retargeter.scp_retarget(
            q_init, selected_link_positions, selected_link_rotations
        )
        q_out.block_until_ready()
        all_times.append(time.perf_counter() - start_time)
        all_iters.append(iters)
    all_times = np.asarray(all_times)
    all_iters = np.asarray(all_iters)
    print(
        "Average time per full retargeting step (milliseconds): ",
        1000 * np.mean(all_times),
    )
    print(
        "Average time per inner differential IK step (milliseconds): ",
        1000 * np.mean(all_times / all_iters),
    )


if __name__ == "__main__":
    main()
