from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="vrpn_mocap",
                executable="client_node",
                name="vrpn_client_0",
                namespace="vrpn_mocap",
                output="screen",
                emulate_tty=True,
                parameters=[{"server": "192.168.110.119", "port": 3883}],
            ),
            Node(
                package="g1_control",
                executable="mocap_node",
                name="mocap_0",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
