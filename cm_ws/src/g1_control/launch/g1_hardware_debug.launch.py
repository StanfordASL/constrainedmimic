from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="g1_control",
                executable="hardware_interface_node",
                name="hardware_interface_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="g1_control",
                executable="hardware_debugging_node.py",
                name="hardware_debugging_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
