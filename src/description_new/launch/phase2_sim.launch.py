"""Launch the two-location world with the existing HV robot and sensor bridge."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package = get_package_share_directory('description_new')
    gazebo_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    resource_path = os.pathsep.join(filter(None, [
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''), os.path.dirname(package)]))

    arguments = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('spawn_x', default_value='-4.45'),
        DeclareLaunchArgument('spawn_y', default_value='2.35'),
        DeclareLaunchArgument('spawn_z', default_value='0.2'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
    ]
    sim_time = LaunchConfiguration('use_sim_time')
    world_file = os.path.join(package, 'worlds', 'nlp_phase2_lab.sdf')

    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(package, 'launch', 'rsp.launch.py')),
        launch_arguments={'use_sim_time': sim_time, 'use_ros2_control': 'false'}.items())

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={'gz_args': ['-r -v 3 ', world_file],
                          'on_exit_shutdown': 'true'}.items())

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=[
            '-world', 'nlp_phase2_lab', '-topic', '/robot_description', '-name', 'hv_bot',
            '-x', LaunchConfiguration('spawn_x'), '-y', LaunchConfiguration('spawn_y'),
            '-z', LaunchConfiguration('spawn_z'), '-Y', LaunchConfiguration('spawn_yaw')],
        parameters=[{'use_sim_time': sim_time}])

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        parameters=[{'config_file': os.path.join(package, 'config', 'gz_bridge.yaml'),
                     'use_sim_time': sim_time}])

    rviz = Node(
        package='rviz2', executable='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(package, 'rviz', 'pioneer3dx.rviz')],
        parameters=[{'use_sim_time': sim_time}])

    return LaunchDescription(arguments + [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        robot_description, gazebo, spawn, bridge, rviz])
