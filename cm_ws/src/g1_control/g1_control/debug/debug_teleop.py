import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from cbfpy import CBF
import mujoco
import mujoco.viewer

from g1_control.rosbag_processing.rosbag_loading import load_ros2_bag
from frax import load_g1
from cm_control.retargeting.retargeting_cbf_configs import (
    JointLimitsFullyActuatedConfig,
)
from cm_control.config.retargeting_config import G1_RETARGETING_IDXS
from cm_control.assets import G1_XML
from cm_control.utils.mujoco_utils import visualize_frame
from cm_control.retargeting.pico_to_g1_retarget import (
    PicoToG1Retargeter,
    StatefulRetargeter,
    RetargeterState,
)

jax.config.update("jax_enable_x64", True)


# bag_path = "/home/dmorton/wboscbf_hardware_ws/bags/rosbag2_2026_02_16-15_40_23"
ws_root = Path(__file__).parents[4]
bag_path = ws_root / "bags" / "rosbag2_2026_04_24-15_05_01_walk"


def main():
    data = load_ros2_bag(bag_path)
    robot = load_g1()
    cbf_config = JointLimitsFullyActuatedConfig(robot)
    cbf = CBF.from_config(cbf_config)
    g1_idxs = jnp.asarray(G1_RETARGETING_IDXS)

    stepsize = 0.2
    max_iters = 20
    convergence_tol = 1e-3

    # TODO rework how these are constructed
    inner_retargeter = PicoToG1Retargeter(
        cbf,
        robot,
        stepsize=stepsize,
        convergence_tol=convergence_tol,
        max_iters=max_iters,
    )
    retargeter = StatefulRetargeter(inner_retargeter)

    n_timesteps = len(data["/xrobo/state"])
    replay_at_real_time = False
    add_debugging_frames = True

    # Set up mujoco viewer
    model_path = str(G1_XML)
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    dummy_retarget_state = RetargeterState.initial()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for i in range(1, n_timesteps):
                start_time = time.time()
                prev_msg_timestamp_ns = data["/xrobo/state"][i - 1]["timestamp"]
                msg_data = data["/xrobo/state"][i]
                dt_ns = msg_data["timestamp"] - prev_msg_timestamp_ns
                # Note that this dt is based on the message frequency (~75 hz)
                # and not the control frequency (50 or 100 hz)
                dt = dt_ns / 1e9

                new_body_poses = np.reshape(msg_data["flat_body_poses"], (24, 7))

                (
                    last_q,
                    last_pos,
                    last_quat_wxyz,
                    last_vel,
                    last_omega,
                    last_contact_mode,
                ) = retargeter.step(new_body_poses, dt)

                d.qpos[7:] = last_q[6:]
                d.qpos[:3] = last_pos
                d.qpos[3:7] = last_quat_wxyz
                mujoco.mj_forward(m, d)

                # Visualization

                # TODO: also be able to visualize the contact mode

                # NOTE: right now this is rather slow because things are running outside JIT
                # But, it's kinda nice to see things in slow-motion

                if add_debugging_frames:
                    viewer.user_scn.ngeom = 0

                    # Update the frames to deal with the height adjustment
                    delta_z = np.array([[0, 0, last_pos[2] - last_q[2]]])

                    # Apply same preprocessing as during teleop so we can visualize what's happening
                    # Note: using a dummy state because all we care about is how the pico poses are processed
                    scaled_target_pos, scaled_target_rmats = (
                        retargeter.functional_retargeter._preprocess_step(
                            new_body_poses, dummy_retarget_state, dt
                        )[:2]
                    )
                    scaled_target_pos += delta_z
                    print(f"Contact mode: {last_contact_mode}")
                    visualize_frame(
                        viewer,
                        scaled_target_pos,
                        scaled_target_rmats,
                        alpha=0.2,
                    )
                    # 3. Robot link frames
                    link_tfs = robot.link_to_world_transforms(last_q)
                    selected_link_pos = link_tfs[g1_idxs, :3, 3]
                    selected_link_rmats = link_tfs[g1_idxs, :3, :3]
                    selected_link_pos += delta_z
                    visualize_frame(
                        viewer,
                        selected_link_pos,
                        selected_link_rmats,
                    )

                viewer.sync()

                elapsed = time.time() - start_time
                if replay_at_real_time and elapsed < dt:
                    time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
