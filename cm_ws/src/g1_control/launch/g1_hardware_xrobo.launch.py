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
                package="xrobo_ros2",
                executable="xrobo_node",
                name="xrobo_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {"adjust_arm_rotations": True},
                ],
            ),
            Node(
                package="g1_control",
                executable="teleop_visualizer_node",
                name="teleop_visualizer_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
