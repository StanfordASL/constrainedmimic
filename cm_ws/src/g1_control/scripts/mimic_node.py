#!/usr/bin/env python3
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from g1_control_msgs.msg import MimicObservation
from cm_control.twist2_utils.motion_lib_twist2 import MotionLib
from cm_control.twist2_utils.twist2_utils import build_mimic_obs_from_motion_lib
from cm_control.twist2_utils.twist2_config import default_mimic_obs
from cm_control.assets import TWIST2_DIR


class MimicNode(Node):
    HOLD_INITIAL_STAND = 0
    RUN_MOTION = 1
    TRANSITION_TO_STAND = 2
    HOLD_FINAL_STAND = 3

    def __init__(self, motion_file):
        super().__init__("mimic_node")

        self.declare_parameter("control_dt", 0.02)
        self.declare_parameter("replay_at_end", True)
        self.control_dt = self.get_parameter("control_dt").value
        self.replay_at_end = self.get_parameter("replay_at_end").value

        # Load Motion Library (Registered as static)
        self.get_logger().info(f"Loading motion file: {motion_file}")
        self.motion_lib = MotionLib(motion_file)

        self.tar_motion_steps = np.array([1])
        self.num_steps = int(self.motion_lib._motion_num_frames[0])

        self.t_step = 0
        self.state = self.HOLD_INITIAL_STAND
        self.transition_timer = 0
        self.transition_duration = 0.5  # seconds
        self.hold_duration = 2.0  # seconds
        self.default_mimic_obs = np.asarray(default_mimic_obs)
        self.last_mimic_obs = self.default_mimic_obs

        self.mimic_obs_pub = self.create_publisher(
            MimicObservation,
            "/g1_control/mimic_obs",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        self.timer = self.create_timer(self.control_dt, self.timer_callback)
        self.get_logger().info("Jit compiling mimic node...")
        self._jit_compile()
        self.get_logger().info(
            "Jit compilation complete\n"
            + "Mimic node initialized. Starting initial stand..."
        )

    def _jit_compile(self):
        num_warmup = 5
        for t_step in range(num_warmup):
            obs = self._jit_build_mimic_obs(t_step)
            obs.block_until_ready()

    @partial(jax.jit, static_argnums=(0,))
    def _jit_build_mimic_obs(self, t_step):
        return build_mimic_obs_from_motion_lib(
            self.motion_lib, t_step, self.control_dt, self.tar_motion_steps
        )

    def _interpolate(self, alpha, obs0, obs1):
        return (1.0 - alpha) * np.asarray(obs0) + alpha * np.asarray(obs1)

    def timer_callback(self):
        obs = None

        if self.state == self.HOLD_INITIAL_STAND:
            obs = self.default_mimic_obs
            self.transition_timer += 1
            if self.transition_timer >= self.hold_duration / self.control_dt:
                self.get_logger().info("Motion beginning...")
                self.state = self.RUN_MOTION
                self.transition_timer = 0
        elif self.state == self.RUN_MOTION:
            # Note: obs is a jax array here. We don't need to convert it to numpy right now
            # because we'll need it as a list for the message constructor
            obs = self._jit_build_mimic_obs(self.t_step)
            self.t_step += 1
            if self.t_step >= self.num_steps:
                self.get_logger().info("Motion ended. Transitioning to final stand...")
                self.state = self.TRANSITION_TO_STAND
                self.transition_timer = 0
                # When we store it though, we can convert once to numpy array and use
                # numpy operations in the interpolate command
                self.last_mimic_obs = np.asarray(obs)

        elif self.state == self.TRANSITION_TO_STAND:
            alpha = self.transition_timer / (self.transition_duration / self.control_dt)
            obs = self._interpolate(alpha, self.last_mimic_obs, self.default_mimic_obs)
            self.transition_timer += 1
            if alpha >= 1.0:
                self.get_logger().info("Holding final stand pose...")
                self.state = self.HOLD_FINAL_STAND
                self.transition_timer = 0

        elif self.state == self.HOLD_FINAL_STAND:
            obs = self.default_mimic_obs
            self.transition_timer += 1
            if (
                self.replay_at_end
                and self.transition_timer >= self.hold_duration / self.control_dt
            ):
                self.get_logger().info("Motion restarting from the beginning...")
                self.state = self.HOLD_INITIAL_STAND
                self.transition_timer = 0
                self.t_step = 0

        if obs is not None:
            msg = MimicObservation()
            msg.data = obs.tolist()
            self.mimic_obs_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--motion_file", type=str, required=True)
    # args_parsed, _ = parser.parse_known_args()

    # node = G1JaxTrajNode(args_parsed.motion_file)
    node = MimicNode(
        str(TWIST2_DIR / "assets/example_motions/0807_yanjie_walk_001.pkl")
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
