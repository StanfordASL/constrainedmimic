from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("add_kinematic_cbf", default_value="false"),
            DeclareLaunchArgument("add_dynamic_cbf", default_value="false"),
            # For karate chop, we can use our simple double-support root state
            # estimator. Other options include "mimic" for using the root state
            # from the reference motion, or "topic" if in sim or using mocap
            DeclareLaunchArgument("root_state_source", default_value="estimator"),
            Node(
                package="g1_control",
                executable="hardware_interface_node",
                name="hardware_interface_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="g1_control",
                executable="karate_chop_node.py",
                name="karate_chop_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "add_kinematic_cbf": LaunchConfiguration("add_kinematic_cbf"),
                        "add_dynamic_cbf": LaunchConfiguration("add_dynamic_cbf"),
                        "root_state_source": LaunchConfiguration("root_state_source"),
                    }
                ],
            ),
        ]
    )
