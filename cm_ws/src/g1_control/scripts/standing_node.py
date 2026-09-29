#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from g1_control_msgs.msg import MimicObservation
from cm_control.twist2_utils.twist2_config import default_mimic_obs


class StandingNode(Node):
    def __init__(self):
        super().__init__("standing_node")

        self.declare_parameter("control_dt", 0.02)
        self.control_dt = self.get_parameter("control_dt").value

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

        # The standing message never changes, we can cache
        self.msg = MimicObservation()
        self.msg.data = default_mimic_obs.tolist()

        self.timer = self.create_timer(self.control_dt, self.timer_callback)
        self.get_logger().info("Standing Node Started")

    def timer_callback(self):
        self.mimic_obs_pub.publish(self.msg)


def main(args=None):
    rclpy.init(args=args)

    node = StandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
