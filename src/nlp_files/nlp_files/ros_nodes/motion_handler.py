#!/usr/bin/env python3
"""Execute manual motion and relay Nav2 through one /cmd_vel publisher."""

import json
import math

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Bool, String

from nlp_files.config import (
    COMMAND_TOPIC,
    MAX_ANGLE_DEG,
    MAX_DISTANCE_M,
    MAX_SPEED_MPS,
    RESULT_TOPIC,
)


class MotionHandler(Node):
    """Convert validated motion frames into closed-loop velocity commands."""

    def __init__(self):
        super().__init__("motion_handler")

        self.declare_parameter("nlp_result_topic", RESULT_TOPIC)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter(
            "execution_result_topic",
            "/nlp/execution_result",
        )
        self.declare_parameter("safety_stop_topic", "/nlp/safety_stop")
        self.declare_parameter("default_linear_speed", 0.15)
        self.declare_parameter("default_angular_speed", 0.5)
        self.declare_parameter("distance_tolerance_m", 0.02)
        self.declare_parameter("angle_tolerance_deg", 2.0)
        self.declare_parameter("odom_timeout_s", 1.0)
        self.declare_parameter("navigation_cmd_vel_topic", "/nlp/nav_cmd_vel")
        self.declare_parameter("navigation_timeout_s", 0.5)

        self.default_linear_speed = float(
            self.get_parameter("default_linear_speed").value
        )
        self.default_angular_speed = float(
            self.get_parameter("default_angular_speed").value
        )
        self.distance_tolerance_m = float(
            self.get_parameter("distance_tolerance_m").value
        )
        self.angle_tolerance_rad = math.radians(
            float(self.get_parameter("angle_tolerance_deg").value)
        )
        self.odom_timeout_s = float(
            self.get_parameter("odom_timeout_s").value
        )

        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.last_odom_time = None
        self.active_command = None
        self.safety_blocked = False
        self.requested_location = None
        self.navigation_location = None
        self.navigation_velocity = None
        self.last_navigation_time = None
        self.stop_latched = False
        self.latest_text = None
        self.navigation_timeout_s = float(
            self.get_parameter("navigation_timeout_s").value
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.get_parameter("cmd_vel_topic").value,
            10,
        )
        self.execution_result_publisher = self.create_publisher(
            String,
            self.get_parameter("execution_result_topic").value,
            10,
        )
        self.cancel_navigation = self.create_client(
            CancelGoal, "/navigate_to_pose/_action/cancel_goal"
        )
        self.navigation_subscription = self.create_subscription(
            Twist, self.get_parameter("navigation_cmd_vel_topic").value,
            self.navigation_velocity_callback, 10,
        )
        self.text_subscription = self.create_subscription(
            String, COMMAND_TOPIC, self.new_text_callback, 10,
        )
        self.execution_subscription = self.create_subscription(
            String, self.get_parameter("execution_result_topic").value,
            self.navigation_result_callback, 10,
        )

        safety_qos = QoSProfile(depth=1)
        safety_qos.reliability = ReliabilityPolicy.RELIABLE
        safety_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.odom_subscription = self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.nlp_result_subscription = self.create_subscription(
            String,
            self.get_parameter("nlp_result_topic").value,
            self.nlp_result_callback,
            10,
        )
        self.safety_stop_subscription = self.create_subscription(
            Bool,
            self.get_parameter("safety_stop_topic").value,
            self.safety_stop_callback,
            safety_qos,
        )
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            "Motion handler ready: manual motion and Nav2 -> one /cmd_vel output"
        )

    def disarm_navigation(self, cancel=False):
        """Prevent an old Nav2 goal from resuming after STOP or manual motion."""
        self.requested_location = None
        self.navigation_location = None
        self.navigation_velocity = None
        self.last_navigation_time = None
        if cancel and self.cancel_navigation.service_is_ready():
            # An empty goal ID and stamp request cancellation of all Nav2 goals.
            future = self.cancel_navigation.call_async(CancelGoal.Request())
            future.add_done_callback(self.navigation_cancelled)

    def navigation_cancelled(self, future):
        try:
            response = future.result()
            if response.return_code not in (0, 2):
                self.get_logger().warn("Nav2 did not confirm cancellation; its velocities remain blocked.")
        except Exception as error:
            self.get_logger().warn(f"Nav2 cancellation failed; its velocities remain blocked: {error}")

    def navigation_velocity_callback(self, message):
        """Cache Nav2 output; the control timer decides whether to forward it."""
        x, z = float(message.linear.x), float(message.angular.z)
        if not math.isfinite(x) or not math.isfinite(z):
            self.navigation_velocity = None
            return
        self.navigation_velocity = (
            max(-MAX_SPEED_MPS, min(MAX_SPEED_MPS, x)),
            max(-0.8, min(0.8, z)),
        )
        self.last_navigation_time = self.get_clock().now()

    def new_text_callback(self, message):
        self.latest_text = message.data
        self.stop_latched = False

    def navigation_result_callback(self, message):
        try:
            result = json.loads(message.data)
            if result.get("handler") != "NAVIGATION_HANDLER":
                return
            if result.get("status") == "STARTED":
                if self.stop_latched:
                    self.disarm_navigation(cancel=True)
                    self.publish_velocity()
                    return
                self.navigation_location = result.get("data", {}).get("location")
            elif result.get("status") in {
                "COMPLETED", "INTERRUPTED", "FAILED", "REJECTED", "STOP_REQUESTED", "IDLE",
            }:
                self.disarm_navigation()
                if self.active_command is None:
                    self.publish_velocity()
        except (ValueError, AttributeError, TypeError):
            return

    def relay_navigation(self):
        """Only an accepted named goal with fresh odometry may drive the robot."""
        now = self.get_clock().now()
        ready = (
            not self.safety_blocked
            and self.requested_location is not None
            and self.requested_location == self.navigation_location
            and self.navigation_velocity is not None
            and self.last_navigation_time is not None
            and self.last_odom_time is not None
        )
        if ready:
            nav_age = (now - self.last_navigation_time).nanoseconds / 1e9
            odom_age = (now - self.last_odom_time).nanoseconds / 1e9
            if 0.0 <= nav_age <= self.navigation_timeout_s and 0.0 <= odom_age <= self.odom_timeout_s:
                self.publish_velocity(*self.navigation_velocity)
                return
        self.publish_velocity()

    def odom_callback(self, message):
        """Cache position and convert the odometry quaternion to yaw."""
        self.current_x = float(message.pose.pose.position.x)
        self.current_y = float(message.pose.pose.position.y)

        orientation = message.pose.pose.orientation
        siny_cosp = 2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y
            + orientation.z * orientation.z
        )
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.last_odom_time = self.get_clock().now()

    def safety_stop_callback(self, message):
        """Stop active motion and block new motion while safety is active."""
        new_state = bool(message.data)
        state_changed = new_state != self.safety_blocked
        self.safety_blocked = new_state

        if self.safety_blocked:
            self.disarm_navigation()
            self.publish_velocity()
        if state_changed:
            self.stop_latched = True

        if self.safety_blocked and self.active_command is not None:
            intent = self.active_command["intent"]
            self.stop_robot()
            self.publish_execution_result(
                "INTERRUPTED",
                intent,
                "Motion stopped by an active safety rule.",
            )

        if state_changed:
            state = "active" if self.safety_blocked else "clear"
            self.get_logger().warn(f"Safety motion block is {state}.")


    def nlp_result_callback(self, message):
        """Accept only valid, single-action frames routed to this handler."""
        try:
            data = json.loads(message.data)
        except (json.JSONDecodeError, TypeError) as error:
            self.get_logger().error(f"Invalid NLP result JSON: {error}")
            return

        if not isinstance(data, dict):
            self.get_logger().error("NLP result must be a JSON object.")
            return
        if data.get("validation", {}).get("decision") != "VALID":
            return

        # A result already being generated when :stop was pressed must not restart motion.
        is_stop = (data.get("route") == "MOTION_HANDLER"
                   and data.get("semantic_frame", {}).get("intent") == "STOP")
        if not is_stop and "text" in data:
            if self.stop_latched or (self.latest_text is not None and data["text"] != self.latest_text):
                return

        if data.get("route") == "NAVIGATION_HANDLER":
            frame = data.get("semantic_frame", {})
            if not isinstance(frame, dict) or not isinstance(frame.get("parameters"), dict):
                return
            location = frame["parameters"].get("location")
            if (frame.get("intent") == "NAVIGATE_TO_LOCATION"
                    and location in {"HOME", "LABORATORY"} and not self.safety_blocked):
                if self.active_command is not None:
                    intent = self.active_command["intent"]
                    self.stop_robot()
                    self.publish_execution_result("INTERRUPTED", intent, "Manual motion stopped for navigation.")
                self.requested_location = location
                self.navigation_velocity = None
            return
        if data.get("route") != "MOTION_HANDLER":
            return

        frame = data.get("semantic_frame", {})
        if not isinstance(frame, dict):
            self.publish_execution_result(
                "REJECTED",
                "UNKNOWN",
                "Motion frame is malformed.",
            )
            return

        intent = frame.get("intent")
        parameters = frame.get("parameters", {})
        if not isinstance(parameters, dict):
            self.publish_execution_result(
                "REJECTED",
                str(intent),
                "Motion parameters are malformed.",
            )
            return

        if intent == "STOP":
            self.stop_latched = True
            self.disarm_navigation(cancel=True)
            self.stop_robot()
            self.publish_execution_result(
                "COMPLETED",
                intent,
                "The robot has been stopped.",
            )
            return


        if self.safety_blocked:
            self.stop_robot()
            self.publish_execution_result(
                "REJECTED",
                str(intent),
                "Motion is blocked by an active safety rule.",
            )
            return

        if self.active_command is not None:
            self.publish_execution_result(
                "REJECTED",
                str(intent),
                "Another motion command is currently being executed.",
            )
            return

        if (
            self.current_x is None
            or self.current_y is None
            or self.current_yaw is None
        ):
            self.publish_execution_result(
                "FAILED",
                str(intent),
                "Motion cannot start because odometry is unavailable.",
            )
            return

        if intent == "MOVE_DISTANCE":
            self.disarm_navigation(cancel=True)
            self.start_linear_motion(parameters)
        elif intent == "TURN_ANGLE":
            self.disarm_navigation(cancel=True)
            self.start_rotation(parameters)
        else:
            self.publish_execution_result(
                "REJECTED",
                str(intent),
                f"Unsupported motion intent: {intent}",
            )

    def start_linear_motion(self, parameters):
        """Validate runtime values and initialise one linear command."""
        direction = parameters.get("direction")
        requested_speed = parameters.get("speed_mps")

        try:
            distance = float(parameters.get("distance_m"))
            speed = (
                self.default_linear_speed
                if requested_speed is None
                else float(requested_speed)
            )
        except (TypeError, ValueError):
            self.publish_execution_result(
                "REJECTED",
                "MOVE_DISTANCE",
                "Distance or speed is invalid.",
            )
            return

        if direction not in {"FORWARD", "BACKWARD"}:
            self.publish_execution_result(
                "REJECTED",
                "MOVE_DISTANCE",
                "Movement direction must be FORWARD or BACKWARD.",
            )
            return
        if not math.isfinite(distance) or not 0.0 < distance <= MAX_DISTANCE_M:
            self.publish_execution_result(
                "REJECTED",
                "MOVE_DISTANCE",
                f"Distance must be within 0 and {MAX_DISTANCE_M} metres.",
            )
            return
        if not math.isfinite(speed) or not 0.0 < speed <= MAX_SPEED_MPS:
            self.publish_execution_result(
                "REJECTED",
                "MOVE_DISTANCE",
                f"Speed must be within 0 and {MAX_SPEED_MPS} m/s.",
            )
            return

        signed_speed = speed if direction == "FORWARD" else -speed
        self.active_command = {
            "type": "linear_motion",
            "intent": "MOVE_DISTANCE",
            "target_distance_m": distance,
            "linear_speed": signed_speed,
            "start_x": self.current_x,
            "start_y": self.current_y,
        }
        self.publish_execution_result(
            "STARTED",
            "MOVE_DISTANCE",
            f"Moving {direction.lower()} for {distance:.2f} metres.",
        )

    def start_rotation(self, parameters):
        """Validate runtime values and initialise one rotation command."""
        direction = parameters.get("direction")
        try:
            angle_deg = float(parameters.get("angle_deg"))
        except (TypeError, ValueError):
            self.publish_execution_result(
                "REJECTED",
                "TURN_ANGLE",
                "Turning angle is invalid.",
            )
            return

        if direction not in {"LEFT", "RIGHT"}:
            self.publish_execution_result(
                "REJECTED",
                "TURN_ANGLE",
                "Turning direction must be LEFT or RIGHT.",
            )
            return
        if not math.isfinite(angle_deg) or not 0.0 < angle_deg <= MAX_ANGLE_DEG:
            self.publish_execution_result(
                "REJECTED",
                "TURN_ANGLE",
                f"Angle must be within 0 and {MAX_ANGLE_DEG} degrees.",
            )
            return
        if (
            not math.isfinite(self.default_angular_speed)
            or self.default_angular_speed <= 0.0
        ):
            self.publish_execution_result(
                "FAILED",
                "TURN_ANGLE",
                "Configured angular speed must be positive.",
            )
            return

        signed_speed = (
            self.default_angular_speed
            if direction == "LEFT"
            else -self.default_angular_speed
        )
        self.active_command = {
            "type": "rotation",
            "intent": "TURN_ANGLE",
            "target_angle_rad": math.radians(angle_deg),
            "angular_speed": signed_speed,
            "last_yaw": self.current_yaw,
            "turned_angle_rad": 0.0,
        }
        self.publish_execution_result(
            "STARTED",
            "TURN_ANGLE",
            f"Turning {direction.lower()} through {angle_deg:.1f} degrees.",
        )

    @staticmethod
    def normalize_angle(angle):
        """Wrap an angle difference into [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def control_loop(self):
        """Advance the active command without blocking ROS callbacks."""
        if self.active_command is None:
            self.relay_navigation()
            return

        command = self.active_command
        intent = command["intent"]

        if self.safety_blocked:
            self.stop_robot()
            self.publish_execution_result(
                "INTERRUPTED",
                intent,
                "Motion stopped by an active safety rule.",
            )
            return

        if self.last_odom_time is None:
            self.stop_robot()
            self.publish_execution_result(
                "FAILED",
                intent,
                "Motion stopped because odometry is unavailable.",
            )
            return

        odom_age_s = (
            self.get_clock().now() - self.last_odom_time
        ).nanoseconds / 1e9
        if odom_age_s < 0.0 or odom_age_s > self.odom_timeout_s:
            self.stop_robot()
            self.publish_execution_result(
                "FAILED",
                intent,
                "Motion stopped because odometry became unresponsive.",
            )
            return

        if command["type"] == "linear_motion":
            travelled_distance = math.hypot(
                self.current_x - command["start_x"],
                self.current_y - command["start_y"],
            )
            tolerance = min(
                self.distance_tolerance_m,
                command["target_distance_m"] * 0.2,
            )
            if travelled_distance + tolerance >= command["target_distance_m"]:
                self.stop_robot()
                self.publish_execution_result(
                    "COMPLETED",
                    intent,
                    f"Completed {travelled_distance:.2f} metres of movement.",
                )
                return

            self.publish_velocity(linear_x=command["linear_speed"])
            return

        if command["type"] == "rotation":
            yaw_change = self.normalize_angle(
                self.current_yaw - command["last_yaw"]
            )
            command["turned_angle_rad"] += abs(yaw_change)
            command["last_yaw"] = self.current_yaw

            if (
                command["turned_angle_rad"] + self.angle_tolerance_rad
                >= command["target_angle_rad"]
            ):
                completed_angle = math.degrees(command["turned_angle_rad"])
                self.stop_robot()
                self.publish_execution_result(
                    "COMPLETED",
                    intent,
                    f"Completed a {completed_angle:.1f}-degree turn.",
                )
                return

            self.publish_velocity(angular_z=command["angular_speed"])
            return

        self.stop_robot()
        self.publish_execution_result(
            "FAILED",
            intent,
            "Unknown active motion type.",
        )

    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        """Publish one velocity command."""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_publisher.publish(twist)

    def stop_robot(self):
        """Clear active state and publish zero velocity."""
        self.active_command = None
        self.publish_velocity()

    def publish_execution_result(
        self,
        status,
        intent,
        response,
        data=None,
    ):
        """Publish one machine-readable execution event."""
        result = {
            "handler": "MOTION_HANDLER",
            "status": status,
            "intent": intent,
            "response": response,
            "data": data if data is not None else {},
        }
        outgoing = String()
        outgoing.data = json.dumps(result, ensure_ascii=False)
        self.execution_result_publisher.publish(outgoing)
        self.get_logger().info(
            f"status={status}, intent={intent}, response={response}"
        )


def main(args=None):
    """Run the motion handler and send a final zero velocity on shutdown."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = MotionHandler()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.disarm_navigation(cancel=True)
                node.stop_robot()
                rclpy.spin_once(node, timeout_sec=0.05)
                node.stop_robot()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
