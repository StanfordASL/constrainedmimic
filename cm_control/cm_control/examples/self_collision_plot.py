"""
Plot for the self collision figure
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from frax import load_g1

from cm_control.utils.assorted_utils import mujoco_qpos_to_q


@jax.jit
def convert_qposes(qposes):
    return jax.vmap(mujoco_qpos_to_q)(qposes)


def main():

    # Load the qpos history
    data_folder = "/home/dmorton/Desktop/Figures/self_collision/data/"
    data_1 = np.load(data_folder + "self_collision_qposes_.npy")
    data_2 = np.load(data_folder + "self_collision_qposes_add_kin_cbf_.npy")
    data_3 = np.load(
        data_folder + "self_collision_qposes_kinematics_only_add_kin_cbf_.npy"
    )

    # Subsample data -- the data from the policy was recorded at every physics timestep
    # which was 1000 hz as opposed to 50 (20x faster)
    # Also, we don't need all of the data recorded (just plot a subset from some start->end)
    # The indices were chosen based on the original plot of all of the data
    idx_start = int(data_3.shape[0] * (2.16 / 12))
    idx_end = int(data_3.shape[0] * (8.33 / 12))
    qs_1 = convert_qposes(data_1[::20])[idx_start:idx_end]
    qs_2 = convert_qposes(data_2[::20])[idx_start:idx_end]
    qs_3 = convert_qposes(data_3)[idx_start:idx_end]

    robot = load_g1()

    @jax.jit
    def compute_sc_dists(qs):
        return jax.vmap(lambda x: jnp.min(robot.self_collision_distances(x)))(qs)

    dists_1 = compute_sc_dists(qs_1)
    dists_2 = compute_sc_dists(qs_2)
    dists_3 = compute_sc_dists(qs_3)

    dt = 1 / 50
    times = np.arange(dists_1.shape[0]) * dt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    ax.set_title("Minimum distance to self-collision")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.plot(times, dists_1, label="Policy +\nUnsafe Reference")
    ax.plot(times, dists_2, label="Policy +\nConstrained Reference")
    ax.plot(times, dists_3, label="Constrained\nReference")
    # Plot a dashed line at y = 0
    ax.plot(times, np.zeros_like(times), "k--")
    # Add a shaded region below y = 0 (unsafe)
    ax.axhspan(ymin=ax.get_ylim()[0], ymax=0, color="red", alpha=0.2)
    # Add legend below the
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
    ax.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
