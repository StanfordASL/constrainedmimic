from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="g1_control",
                executable="sim_node",
                name="g1_sim_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="g1_control",
                executable="hardware_interface_node",
                name="hardware_interface_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="g1_control",
                executable="policy_node.py",
                name="policy_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="g1_control",
                executable="mimic_node.py",
                name="mimic_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
