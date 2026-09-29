from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # Lift the robot off the ground on the virtual gantry so the
            # oscillating joint is free to move
            DeclareLaunchArgument("gantry_height_offset", default_value="0.1"),
            Node(
                package="g1_control",
                executable="sim_node",
                name="g1_sim_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "gantry_height_offset": LaunchConfiguration(
                            "gantry_height_offset"
                        ),
                    }
                ],
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
                executable="hardware_debugging_node.py",
                name="hardware_debugging_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
