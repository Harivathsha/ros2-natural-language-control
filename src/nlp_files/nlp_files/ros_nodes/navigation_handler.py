#!/usr/bin/env python3
"""Send validated HOME/LABORATORY commands to Nav2; record poses using TF.

Run after mapping/localization and Nav2 are working. Stop the robot at each
destination, then call /nlp/save_home or /nlp/save_laboratory (std_srvs/Trigger).
Record both again whenever the map changes. HOME is a parking pose near the
charger, not an automatic docking or battery-charging operation.
"""

import json
import math
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from nlp_files.config import PROJECT_ROOT, RESULT_TOPIC


class NavigationHandler(Node):
    def __init__(self):
        super().__init__("navigation_handler")
        for name, value in {
            "map_frame": "map",
            "base_frame": "base_footprint",
            "locations_file": str(PROJECT_ROOT / "artifacts/named_locations.json"),
        }.items():
            self.declare_parameter(name, value)
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.locations_file = Path(self.get_parameter("locations_file").value).expanduser()
        self.locations = self.load_locations()
        self.active_location = None
        self.goal_handle = None
        self.cancel_requested = False
        self.cancel_in_flight = False
        self.safety_blocked = False

        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.results = self.create_publisher(String, "/nlp/execution_result", 10)
        self.commands = self.create_subscription(String, RESULT_TOPIC, self.on_command, 10)
        safety_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.safety = self.create_subscription(
            Bool, "/nlp/safety_stop", self.on_safety_stop, safety_qos,
        )
        self.home_service = self.create_service(
            Trigger, "/nlp/save_home",
            lambda request, response: self.save_location("HOME", response),
        )
        self.lab_service = self.create_service(
            Trigger, "/nlp/save_laboratory",
            lambda request, response: self.save_location("LABORATORY", response),
        )
        self.get_logger().info(
            f"Navigation handler ready. Saved locations: {', '.join(self.locations) or 'none'}. "
            "Nav2 and a localized map are required."
        )

    def load_locations(self):
        if not self.locations_file.exists():
            return {}
        try:
            document = json.loads(self.locations_file.read_text())
            if document["frame_id"] != self.map_frame:
                raise ValueError("Saved locations use a different map frame.")
            locations = document["locations"]
            if not isinstance(locations, dict) or set(locations) - {"HOME", "LABORATORY"}:
                raise ValueError("Only HOME and LABORATORY are supported.")
            for pose in locations.values():
                if not isinstance(pose, dict) or set(pose) != {"x", "y", "yaw"}:
                    raise ValueError("Each location needs x, y and yaw.")
                if any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in pose.values()):
                    raise ValueError("Location coordinates must be finite numbers.")
            return locations
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid locations file {self.locations_file}: {error}") from error

    def current_pose(self):
        transform = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        age = (self.get_clock().now().nanoseconds
               - Time.from_msg(transform.header.stamp).nanoseconds) / 1e9
        if abs(age) > 2.0:
            raise ValueError("The map-to-robot transform is stale. Check localization and simulation time.")
        p, q = transform.transform.translation, transform.transform.rotation
        values = (p.x, p.y, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values) or sum(v*v for v in values[2:]) < 0.5:
            raise ValueError("The robot pose is invalid.")
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        return {"x": p.x, "y": p.y, "yaw": yaw}

    def save_location(self, location, response):
        if self.active_location is not None:
            response.message = "Finish or stop navigation before recording a location."
            return response
        try:
            pose = self.current_pose()
            updated = {**self.locations, location: pose}
            self.locations_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.locations_file.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(
                {"frame_id": self.map_frame, "locations": updated}, indent=2,
            ) + "\n")
            temporary.replace(self.locations_file)
            self.locations = updated
            response.success = True
            response.message = f"Saved {location}: {pose}. Record again if you change the map."
            self.get_logger().info(response.message)
        except (TransformException, ValueError, OSError) as error:
            response.message = f"Could not record {location}: {error}"
        return response

    def report(self, status, response, intent="NAVIGATE_TO_LOCATION", location=None):
        data = {"location": location or self.active_location}
        self.results.publish(String(data=json.dumps({
            "handler": "NAVIGATION_HANDLER", "status": status,
            "intent": intent, "response": response, "data": data,
        })))
        self.get_logger().info(f"{status}: {response}")

    def on_command(self, message):
        try:
            data = json.loads(message.data)
            if data.get("validation", {}).get("decision") != "VALID":
                return
            frame = data["semantic_frame"]
            if frame.get("intent") == "STOP" and data.get("route") == "MOTION_HANDLER":
                self.request_stop()
                return
            if data.get("route") != "NAVIGATION_HANDLER":
                return
            if (frame.get("intent") != "NAVIGATE_TO_LOCATION"
                    or frame.get("capability") != "NAVIGATION_COMMAND"):
                raise ValueError("Expected a named navigation frame.")
            parameters = frame["parameters"]
            if set(parameters) != {"location"} or not isinstance(parameters["location"], str):
                raise ValueError("Navigation needs exactly one named location.")
            location = parameters["location"]
            if location not in {"HOME", "LABORATORY"}:
                raise ValueError("Choose HOME or LABORATORY.")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.report("REJECTED", f"Invalid NLP navigation result: {error}")
            return
        if self.safety_blocked:
            self.report("REJECTED", "Navigation is blocked by an active safety rule.", location=location)
            return
        if self.active_location is not None:
            self.report("REJECTED", "Navigation is busy. Send stop before another destination.", location=location)
            return
        if location not in self.locations:
            self.report("REJECTED", f"Record {location} first using its save service.", location=location)
            return
        if not self.nav.server_is_ready():
            self.report("REJECTED", "Nav2 /navigate_to_pose is unavailable. Start Nav2 first.", location=location)
            return
        try:
            self.current_pose()  # Require a fresh map-to-robot transform.
        except (TransformException, ValueError) as error:
            self.report("REJECTED", f"Localization is not ready: {error}", location=location)
            return

        pose = self.locations[location]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(pose["x"])
        goal.pose.pose.position.y = float(pose["y"])
        goal.pose.pose.orientation.z = math.sin(pose["yaw"] / 2)
        goal.pose.pose.orientation.w = math.cos(pose["yaw"] / 2)
        self.active_location = location
        self.cancel_requested = False
        try:
            self.nav.send_goal_async(goal).add_done_callback(self.on_goal_response)
        except Exception as error:
            self.report("FAILED", f"Could not send navigation goal: {error}")
            self.clear_goal()

    def on_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as error:
            self.report("FAILED", f"Nav2 goal response failed: {error}")
            self.clear_goal()
            return
        if not handle.accepted:
            self.report("REJECTED", "Nav2 rejected the navigation goal.")
            self.clear_goal()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.on_result)
        self.report("STARTED", f"Navigating to {self.active_location}.")
        if self.cancel_requested:
            self.cancel_goal()

    def on_result(self, future):
        try:
            result = future.result()
        except Exception as error:
            self.report("FAILED", f"Navigation result unavailable; send stop before restarting: {error}")
            return  # Keep busy: execution has not been confirmed finished.
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.report("COMPLETED", f"Arrived at {self.active_location}.")
        elif result.status == GoalStatus.STATUS_CANCELED:
            self.report("INTERRUPTED", "Nav2 confirmed that navigation was canceled.")
        elif result.status == GoalStatus.STATUS_ABORTED:
            self.report("FAILED", f"Nav2 aborted navigation (error {getattr(result.result, 'error_code', 'unknown')}).")
        else:
            self.report("FAILED", f"Unexpected Nav2 result status: {result.status}.")
            return  # An unexpected status does not confirm a finished goal.
        self.clear_goal()

    def request_stop(self):
        if self.active_location is None:
            self.report("IDLE", "No navigation goal is active.", intent="STOP")
            return
        self.cancel_requested = True
        self.report("STOP_REQUESTED", "Requesting Nav2 cancellation.", intent="STOP")
        if self.goal_handle is not None:
            self.cancel_goal()

    def cancel_goal(self):
        if self.cancel_in_flight:
            return
        handle = self.goal_handle
        self.cancel_in_flight = True
        try:
            handle.cancel_goal_async().add_done_callback(
                lambda future: self.on_cancel_response(future, handle),
            )
        except Exception as error:
            self.cancel_in_flight = False
            self.report("FAILED", f"Could not request cancellation: {error}", intent="STOP")

    def on_cancel_response(self, future, handle):
        if self.goal_handle is not handle:
            return
        self.cancel_in_flight = False
        try:
            if not future.result().goals_canceling:
                self.report("FAILED", "Nav2 did not accept cancellation; check Nav2 status.", intent="STOP")
        except Exception as error:
            self.report("FAILED", f"Cancellation response unavailable: {error}", intent="STOP")
        # Remain busy until the action result confirms completion/cancellation.

    def on_safety_stop(self, message):
        self.safety_blocked = bool(message.data)
        if self.safety_blocked and self.active_location is not None:
            self.request_stop()

    def clear_goal(self):
        self.active_location = None
        self.goal_handle = None
        self.cancel_requested = False
        self.cancel_in_flight = False


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NavigationHandler()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok() and node.active_location is not None:
                node.request_stop()
                deadline = time.monotonic() + 2.0
                try:
                    while node.active_location is not None and time.monotonic() < deadline:
                        rclpy.spin_once(node, timeout_sec=0.1)
                except (KeyboardInterrupt, ExternalShutdownException):
                    pass
                if node.active_location is not None:
                    node.get_logger().warning("Nav2 cancellation was not confirmed before exit.")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
