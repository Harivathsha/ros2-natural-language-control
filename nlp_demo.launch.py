"""Terminal 2: learned NLP + handlers + mapping/localization + Nav2 + RViz.

Keep Gazebo in terminal 1 and robot_ui or ros_command_client in terminal 3. Run this file
from the nlp_project root, with the robotics_ai environment activated.
"""

import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
    OpaqueFunction, RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def prepare_nav2(source):
    """Use the existing DWB settings, adding costmaps for this robot/world."""
    settings = yaml.safe_load(source.read_text())
    required = (
        "controller_server", "planner_server", "smoother_server",
        "behavior_server", "bt_navigator", "velocity_smoother",
    )
    for name in required:
        if name not in settings:
            raise RuntimeError(f"{source} is missing {name}.")
        settings[name]["ros__parameters"]["use_sim_time"] = True

    controller = settings["controller_server"]["ros__parameters"]
    controller["enable_stamped_cmd_vel"] = False
    controller["min_y_velocity_threshold"] = 0.001
    controller["progress_checker"].update(
        required_movement_radius=0.15, movement_time_allowance=30.0,
    )
    controller["general_goal_checker"].update(
        xy_goal_tolerance=0.18, yaw_goal_tolerance=0.20,
    )
    follow = controller["FollowPath"]
    if follow.get("plugin") != "dwb_core::DWBLocalPlanner":
        raise RuntimeError("This demo expects your existing DWB controller settings.")
    follow.update(
        max_vel_x=0.20, max_speed_xy=0.20, max_vel_theta=0.8,
        acc_lim_x=0.5, decel_lim_x=-0.5,
        acc_lim_theta=1.5, decel_lim_theta=-1.5,
        trans_stopped_velocity=0.05, xy_goal_tolerance=0.18,
        debug_trajectory_details=False, publish_cost_grid_pc=False,
    )
    behavior = settings["behavior_server"]["ros__parameters"]
    behavior.update(
        robot_base_frame="base_footprint", enable_stamped_cmd_vel=False,
        max_rotational_vel=0.8, min_rotational_vel=0.2,
    )
    settings["bt_navigator"]["ros__parameters"].update(
        robot_base_frame="base_footprint", global_frame="map", odom_topic="/odom",
    )
    settings["velocity_smoother"]["ros__parameters"].update(
        enable_stamped_cmd_vel=False,
        max_velocity=[0.20, 0.0, 0.8], min_velocity=[-0.20, 0.0, -0.8],
        max_accel=[0.5, 0.0, 1.5], max_decel=[-0.5, 0.0, -1.5],
        velocity_timeout=0.5, odom_topic="/odom",
    )
    obstacle = {
        "plugin": "nav2_costmap_2d::ObstacleLayer", "enabled": True,
        "observation_sources": "scan",
        "scan": {
            "topic": "/scan", "data_type": "LaserScan",
            "clearing": True, "marking": True,
            "min_obstacle_height": 0.03, "max_obstacle_height": 2.0,
            "raytrace_max_range": 8.0, "raytrace_min_range": 0.0,
            "obstacle_max_range": 6.0, "obstacle_min_range": 0.15,
        },
    }
    for name, frame, global_map in (
        ("local_costmap", "odom", False), ("global_costmap", "map", True),
    ):
        params = {
            "use_sim_time": True, "global_frame": frame,
            "robot_base_frame": "base_footprint", "robot_radius": 0.20,
            "resolution": 0.05, "transform_tolerance": 0.5,
            "update_frequency": 5.0 if not global_map else 1.0,
            "publish_frequency": 2.0 if not global_map else 1.0,
            "plugins": (["static_layer"] if global_map else [])
                       + ["obstacle_layer", "inflation_layer"],
            "obstacle_layer": copy.deepcopy(obstacle),
            "inflation_layer": {
                "plugin": "nav2_costmap_2d::InflationLayer",
                "inflation_radius": 0.40, "cost_scaling_factor": 3.0,
            },
            "always_send_full_costmap": True,
        }
        if global_map:
            params.update(track_unknown_space=True, static_layer={
                "plugin": "nav2_costmap_2d::StaticLayer",
                "map_subscribe_transient_local": True,
                "subscribe_to_updates": True,
            })
        else:
            params.update(rolling_window=True, width=3, height=3)
        settings[name] = {name: {"ros__parameters": params}}
    return settings


def saved_home(folder):
    """Check that the saved map and its two locations are a complete pair."""
    map_file, locations_file = folder / "map.yaml", folder / "named_locations.json"
    if not map_file.is_file() or not locations_file.is_file():
        raise RuntimeError("Saved environment is incomplete. Use mapping:=true to create a new map.")
    metadata = yaml.safe_load(map_file.read_text())
    if not isinstance(metadata, dict) or metadata.get("image") != "map.pgm" or not (folder / "map.pgm").is_file():
        raise RuntimeError("Saved map image is missing or invalid.")
    if not isinstance(metadata.get("resolution"), (int, float)) or not math.isfinite(metadata["resolution"]) or metadata["resolution"] <= 0:
        raise RuntimeError("Saved map resolution is invalid.")
    document = json.loads(locations_file.read_text())
    locations = document.get("locations", {})
    if document.get("frame_id") != "map" or set(locations) != {"HOME", "LABORATORY"}:
        raise RuntimeError("Save both HOME and LAB with this map first.")
    for pose in locations.values():
        if (not isinstance(pose, dict) or set(pose) != {"x", "y", "yaw"}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose.values())):
            raise RuntimeError("Saved location coordinates are invalid.")
    return locations["HOME"]


def start_stack(context):
    project = Path(LaunchConfiguration("project_root").perform(context)).expanduser().resolve()
    model = Path(LaunchConfiguration("model_path").perform(context)).expanduser()
    if not model.is_absolute():
        model = project / model
    if not (model / "config.json").is_file():
        raise RuntimeError(f"Trained model not found at {model}.")
    source = project / "src/navigation/config/navigation.yaml"
    if not source.is_file():
        raise RuntimeError(f"Navigation settings not found: {source}")

    # Parameters are temporary; the saved map and its locations stay in artifacts.
    temporary = tempfile.TemporaryDirectory(prefix="nlp_demo_")
    work = Path(temporary.name)
    nav_file = work / "nav2.yaml"
    settings = prepare_nav2(source)
    saved = project / "artifacts/saved_environment"
    previous = saved.with_name("saved_environment.previous")
    if not saved.exists() and previous.is_dir():
        previous.rename(saved)  # Recover an interrupted folder replacement.
    mapping = LaunchConfiguration("mapping").perform(context).lower()
    if mapping not in {"true", "false"}:
        raise RuntimeError("Use mapping:=true or mapping:=false.")
    if saved.exists() and mapping == "false":
        home = saved_home(saved)
        defaults = Path(get_package_share_directory("nav2_bringup")) / "params/nav2_params.yaml"
        amcl = yaml.safe_load(defaults.read_text())["amcl"]["ros__parameters"]
        amcl.update(
            use_sim_time=True, base_frame_id="base_footprint", odom_frame_id="odom",
            global_frame_id="map", scan_topic="/scan", set_initial_pose=True,
            always_reset_initial_pose=True, tf_broadcast=True,
            initial_pose={"x": float(home["x"]), "y": float(home["y"]),
                          "z": 0.0, "yaw": float(home["yaw"])},
        )
        settings["amcl"] = {"ros__parameters": amcl}
        settings["map_server"] = {"ros__parameters": {
            "use_sim_time": True, "yaml_filename": str(saved / "map.yaml"),
            "topic_name": "map", "frame_id": "map",
        }}
        locations_file = saved / "named_locations.json"
        localization = [
            LogInfo(msg="Loading saved map + HOME/LAB. Start the simulation at its usual HOME pose."),
            Node(package="nav2_map_server", executable="map_server", name="map_server",
                 parameters=[str(nav_file)], output="screen"),
            Node(package="nav2_amcl", executable="amcl", name="amcl",
                 parameters=[str(nav_file)], output="screen"),
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                 name="lifecycle_manager_localization", output="screen",
                 parameters=[{"use_sim_time": True, "autostart": True,
                              "node_names": ["map_server", "amcl"]}]),
        ]
    else:
        locations_file = work / "locations.json"
        slam_share = Path(get_package_share_directory("slam_toolbox"))
        slam_source = project / "src/navigation/config/slam_toolbox_mapping.yaml"
        if not slam_source.is_file():
            slam_source = slam_share / "config/mapper_params_online_async.yaml"
        slam = yaml.safe_load(slam_source.read_text())
        slam["slam_toolbox"]["ros__parameters"].update(
            use_sim_time=True, mode="mapping", map_frame="map", odom_frame="odom",
            base_frame="base_footprint", scan_topic="/scan", debug_logging=False,
            map_update_interval=1.0, minimum_travel_distance=0.10,
            minimum_travel_heading=0.10,
        )
        slam_file = work / "slam.yaml"
        slam_file.write_text(yaml.safe_dump(slam))
        localization = [
            LogInfo(msg="Mapping: record HOME and LAB, then click Save map in the UI."),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(slam_share / "launch/online_async_launch.py")),
                launch_arguments={"slam_params_file": str(slam_file),
                                  "use_sim_time": "true", "autostart": "true"}.items(),
            ),
        ]
    nav_file.write_text(yaml.safe_dump(settings))

    rviz = yaml.safe_load((project / "src/navigation/rviz/navigation.rviz").read_text())
    for display in rviz.get("Visualization Manager", {}).get("Displays", []):
        if display.get("Class") == "rviz_default_plugins/RobotModel":
            topic = display.get("Description Topic")
            if isinstance(topic, dict):
                topic["Durability Policy"] = "Transient Local"
    rviz_file = work / "demo.rviz"
    rviz_file.write_text(yaml.safe_dump(rviz))

    venv = os.environ.get("VIRTUAL_ENV")
    python = str(Path(venv) / "bin/python") if venv else sys.executable
    env = {
        "NLP_PROJECT_ROOT": str(project), "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [
            str(project / "src/nlp_files"), os.environ.get("PYTHONPATH", ""),
        ])),
    }

    def nlp_node(module, **parameters):
        args = [python, "-m", "nlp_files.ros_nodes." + module, "--ros-args",
                "-p", "use_sim_time:=true"]
        for key, value in parameters.items():
            args += ["-p", f"{key}:={value}"]
        return ExecuteProcess(cmd=args, cwd=str(project), additional_env=env,
                              output="screen")

    servers = [
        ("nav2_controller", "controller_server"),
        ("nav2_smoother", "smoother_server"),
        ("nav2_planner", "planner_server"),
        ("nav2_behaviors", "behavior_server"),
        ("nav2_bt_navigator", "bt_navigator"),
        ("nav2_velocity_smoother", "velocity_smoother"),
    ]
    actions = [
        RegisterEventHandler(OnShutdown(on_shutdown=[
            OpaqueFunction(function=lambda context: temporary.cleanup()),
        ])),
    ] + localization
    for package, name in servers:
        remaps = [("/tf", "tf"), ("/tf_static", "tf_static")]
        if name in {"controller_server", "behavior_server"}:
            remaps += [("cmd_vel", "/nlp/nav_cmd_vel_raw")]
        elif name == "velocity_smoother":
            remaps += [("cmd_vel", "/nlp/nav_cmd_vel_raw"),
                       ("cmd_vel_smoothed", "/nlp/nav_cmd_vel")]
        actions.append(Node(package=package, executable=name, name=name,
                            parameters=[str(nav_file)], remappings=remaps,
                            output="screen"))
    actions += [
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_navigation", output="screen",
             parameters=[{"use_sim_time": True, "autostart": True,
                          "node_names": [name for _, name in servers]}]),
        nlp_node("nlp_ros_node", model_path=str(model)),
        nlp_node("motion_handler"),
        nlp_node("navigation_handler", locations_file=str(locations_file)),
        nlp_node("diagnosis_handler", camera_topic="/camera/image"),
        nlp_node("safety_rule_handler", camera_topic="/camera/image"),
        Node(package="rviz2", executable="rviz2", output="screen",
             condition=IfCondition(LaunchConfiguration("rviz")),
             arguments=["-d", str(rviz_file)],
             parameters=[{"use_sim_time": True}]),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("project_root", default_value=os.getcwd()),
        DeclareLaunchArgument("model_path", default_value="artifacts/semantic_phase2_typing"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("mapping", default_value="false",
                              description="Force a new mapping session instead of loading the saved environment."),
        OpaqueFunction(function=start_stack),
    ])
