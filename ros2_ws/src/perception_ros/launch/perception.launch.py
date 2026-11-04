"""ros2 launch perception_ros perception.launch.py scenario:=cpna25 oracle_depth:=false  (scenarios: cpna25, cpna75, cpfa50, ccrb, none)"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        "carla_host": EnvironmentVariable("CARLA_HOST", default_value="localhost"),
        "scenario": "cpna25",
        "oracle_depth": "false",
        "spawn_index": "0",
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()],
        Node(package="perception_ros", executable="carla_bridge", name="carla_bridge", output="screen",
             parameters=[{"host": LaunchConfiguration("carla_host"), "scenario": LaunchConfiguration("scenario"),
                          "spawn_index": LaunchConfiguration("spawn_index")}]),
        Node(package="perception_ros", executable="perception", name="perception", output="screen",
             parameters=[{"oracle_depth": LaunchConfiguration("oracle_depth")}]),
        Node(package="perception_ros", executable="aeb", name="aeb", output="screen"),
    ])
