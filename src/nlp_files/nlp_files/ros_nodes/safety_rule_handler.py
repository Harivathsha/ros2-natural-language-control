#!/usr/bin/env python3
"""Store and monitor conditional safety rules created through NLP."""

import json
import math
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Bool, String

from nlp_files.config import RESULT_TOPIC


class SafetyRuleHandler(Node):
    """Manage in-memory rules and publish a fail-safe motion-block state."""

    SENSORS = ("LIDAR", "CAMERA", "IMU")
    PROPERTIES = {
        "MINIMUM_OBSTACLE_DISTANCE_M",
        "PUBLISH_RATE_HZ",
        "MESSAGE_AGE_S",
    }
    OPERATORS = {"<", "<=", ">", ">="}
    ACTIONS = {"STOP", "BLOCK_MOTION", "WARN"}
    PROPERTY_UNITS = {
        "MINIMUM_OBSTACLE_DISTANCE_M": "m",
        "PUBLISH_RATE_HZ": "Hz",
        "MESSAGE_AGE_S": "s",
    }

    def __init__(self):
        super().__init__("safety_rule_handler")

        self.declare_parameter("nlp_result_topic", RESULT_TOPIC)
        self.declare_parameter("lidar_topic", "/scan")
        self.declare_parameter("camera_topic", "/camera/image")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter(
            "execution_result_topic",
            "/nlp/execution_result",
        )
        self.declare_parameter("safety_stop_topic", "/nlp/safety_stop")
        self.declare_parameter("monitor_period_s", 0.1)
        self.declare_parameter("sensor_timeout_s", 2.0)

        self.rules = {}
        self.next_rule_number = 1
        self.safety_blocked = False
        self.sensor_timeout_s = float(
            self.get_parameter("sensor_timeout_s").value
        )
        self.last_received = {sensor: None for sensor in self.SENSORS}
        self.message_times = {
            sensor: deque(maxlen=30) for sensor in self.SENSORS
        }
        self.nearest_obstacle_m = None

        self.execution_result_publisher = self.create_publisher(
            String,
            self.get_parameter("execution_result_topic").value,
            10,
        )

        safety_qos = QoSProfile(depth=1)
        safety_qos.reliability = ReliabilityPolicy.RELIABLE
        safety_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.safety_stop_publisher = self.create_publisher(
            Bool,
            self.get_parameter("safety_stop_topic").value,
            safety_qos,
        )

        self.nlp_result_subscription = self.create_subscription(
            String,
            self.get_parameter("nlp_result_topic").value,
            self.nlp_result_callback,
            10,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            self.get_parameter("lidar_topic").value,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.camera_subscription = self.create_subscription(
            Image,
            self.get_parameter("camera_topic").value,
            self.camera_callback,
            qos_profile_sensor_data,
        )
        self.imu_subscription = self.create_subscription(
            Imu,
            self.get_parameter("imu_topic").value,
            self.imu_callback,
            qos_profile_sensor_data,
        )
        self.monitor_timer = self.create_timer(
            float(self.get_parameter("monitor_period_s").value),
            self.monitor_rules,
        )

        self.publish_safety_state(False)
        self.get_logger().info("Safety rule handler ready.")

    def now_seconds(self):
        """Return current ROS time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def record_message_time(self, sensor):
        """Record one sensor message arrival."""
        current_time = self.now_seconds()
        self.last_received[sensor] = current_time
        self.message_times[sensor].append(current_time)

    def scan_callback(self, message):
        """Cache the nearest finite LiDAR return."""
        finite_ranges = [
            float(value)
            for value in message.ranges
            if math.isfinite(value)
            and message.range_min <= value <= message.range_max
        ]
        self.nearest_obstacle_m = (
            min(finite_ranges) if finite_ranges else math.inf
        )
        self.record_message_time("LIDAR")

    def camera_callback(self, _message):
        """Record camera arrival for rate and freshness rules."""
        self.record_message_time("CAMERA")

    def imu_callback(self, _message):
        """Record IMU arrival for rate and freshness rules."""
        self.record_message_time("IMU")

    def get_message_age(self, sensor):
        """Return message age, treating never-received data as infinite."""
        last_time = self.last_received.get(sensor)
        if last_time is None:
            return math.inf
        return max(0.0, self.now_seconds() - last_time)

    def get_message_rate(self, sensor):
        """Return recent average rate, or zero before enough samples."""
        times = self.message_times.get(sensor)
        if times is None or len(times) < 2:
            return 0.0
        if self.get_message_age(sensor) > self.sensor_timeout_s:
            return 0.0
        duration = times[-1] - times[0]
        if duration <= 0.0:
            return 0.0
        return (len(times) - 1) / duration

    def nlp_result_callback(self, message):
        """Apply valid safety-management frames from the NLP pipeline."""
        try:
            result = json.loads(message.data)
        except (json.JSONDecodeError, TypeError) as error:
            self.get_logger().error(f"Invalid NLP result JSON: {error}")
            return

        if not isinstance(result, dict):
            self.get_logger().error("NLP result must be a JSON object.")
            return
        if result.get("route") != "SAFETY_RULE_HANDLER":
            return
        if result.get("validation", {}).get("decision") != "VALID":
            return

        frame = result.get("semantic_frame", {})
        intent = frame.get("intent")
        parameters = frame.get("parameters", {})
        if not isinstance(parameters, dict):
            self.publish_execution_result(
                "REJECTED",
                str(intent),
                "Safety-rule parameters are malformed.",
            )
            return

        if intent == "CREATE_SAFETY_RULE":
            self.create_rule(parameters)
        elif intent == "LIST_SAFETY_RULES":
            self.list_rules()
        elif intent == "REMOVE_SAFETY_RULE":
            self.remove_rule(parameters.get("rule_id"))
        elif intent == "CLEAR_SAFETY_RULES":
            self.clear_rules()
        else:
            self.publish_execution_result(
                "REJECTED",
                str(intent),
                f"Unsupported safety-rule intent: {intent}",
            )

    def create_rule(self, parameters):
        """Validate and store one conditional rule."""
        sensor = parameters.get("sensor")
        action = parameters.get("action")
        property_name = parameters.get("condition_property")
        comparison = parameters.get("operator")
        connector = parameters.get("connector")
        threshold_unit = parameters.get("threshold_unit")

        try:
            threshold = float(parameters.get("threshold_value"))
        except (TypeError, ValueError):
            threshold = math.nan

        sensor_supported = sensor in self.SENSORS or sensor == "ALL"
        if (
            not sensor_supported
            or action not in self.ACTIONS
            or property_name not in self.PROPERTIES
            or comparison not in self.OPERATORS
            or connector not in {"IF", "WHEN"}
            or threshold_unit != self.PROPERTY_UNITS.get(property_name)
            or not math.isfinite(threshold)
            or threshold <= 0.0
        ):
            self.publish_execution_result(
                "REJECTED",
                "CREATE_SAFETY_RULE",
                "The safety rule contains unsupported runtime values.",
            )
            return

        if property_name == "MINIMUM_OBSTACLE_DISTANCE_M":
            sensor = "LIDAR"

        rule_id = f"R{self.next_rule_number:03d}"
        self.next_rule_number += 1
        self.rules[rule_id] = {
            "rule_id": rule_id,
            "sensor": sensor,
            "action": action,
            "condition_property": property_name,
            "operator": comparison,
            "threshold_value": threshold,
            "threshold_unit": threshold_unit,
            "connector": connector,
            "triggered": False,
        }

        self.publish_execution_result(
            "CREATED",
            "CREATE_SAFETY_RULE",
            f"Safety rule {rule_id} was created.",
            {"rule": self.public_rule(self.rules[rule_id])},
        )

    @staticmethod
    def public_rule(rule):
        """Return a rule without mutable internal monitoring state."""
        return {
            key: value
            for key, value in rule.items()
            if key != "triggered"
        }

    def list_rules(self):
        """Publish all currently stored rules."""
        rule_list = [self.public_rule(rule) for rule in self.rules.values()]
        response = (
            "No safety rules are currently stored."
            if not rule_list
            else f"{len(rule_list)} safety rule(s) are currently stored."
        )
        self.publish_execution_result(
            "COMPLETED",
            "LIST_SAFETY_RULES",
            response,
            {"rules": rule_list},
        )

    def remove_rule(self, rule_id):
        """Remove one rule by its canonical identifier."""
        if rule_id not in self.rules:
            self.publish_execution_result(
                "NOT_FOUND",
                "REMOVE_SAFETY_RULE",
                f"Safety rule {rule_id} was not found.",
            )
            return

        removed = self.public_rule(self.rules.pop(rule_id))
        self.monitor_rules()
        self.publish_execution_result(
            "REMOVED",
            "REMOVE_SAFETY_RULE",
            f"Safety rule {rule_id} was removed.",
            {"rule": removed},
        )

    def clear_rules(self):
        """Remove all rules and clear the safety motion block."""
        removed_count = len(self.rules)
        self.rules.clear()
        self.set_safety_block(False)
        self.publish_execution_result(
            "CLEARED",
            "CLEAR_SAFETY_RULES",
            f"Cleared {removed_count} safety rule(s).",
            {"removed_count": removed_count},
        )

    def metric_for_rule(self, rule):
        """Read the live metric monitored by one rule."""
        property_name = rule["condition_property"]
        sensor = rule["sensor"]

        if property_name == "MINIMUM_OBSTACLE_DISTANCE_M":
            if self.last_received["LIDAR"] is None:
                return math.inf
            return self.nearest_obstacle_m

        selected_sensors = self.SENSORS if sensor == "ALL" else (sensor,)
        if property_name == "PUBLISH_RATE_HZ":
            rates = [self.get_message_rate(name) for name in selected_sensors]
            return min(rates)
        if property_name == "MESSAGE_AGE_S":
            ages = [self.get_message_age(name) for name in selected_sensors]
            return max(ages)
        return None

    @staticmethod
    def compare(value, comparison, threshold):
        """Evaluate one canonical numeric comparison."""
        if value is None:
            return False
        if comparison == "<":
            return value < threshold
        if comparison == "<=":
            return value <= threshold
        if comparison == ">":
            return value > threshold
        if comparison == ">=":
            return value >= threshold
        return False

    def monitor_rules(self):
        """Evaluate all rules and maintain the combined motion-block state."""
        blocking_rule_ids = []

        for rule_id, rule in self.rules.items():
            value = self.metric_for_rule(rule)
            triggered = self.compare(
                value,
                rule["operator"],
                rule["threshold_value"],
            )

            if triggered and not rule["triggered"]:
                printable_value = (
                    "infinity" if math.isinf(value) else round(value, 4)
                )
                self.publish_execution_result(
                    "TRIGGERED",
                    "SAFETY_MONITOR",
                    f"Safety rule {rule_id} triggered action {rule['action']}.",
                    {
                        "rule": self.public_rule(rule),
                        "measured_value": printable_value,
                    },
                )

            rule["triggered"] = triggered
            if triggered and rule["action"] in {"STOP", "BLOCK_MOTION"}:
                blocking_rule_ids.append(rule_id)

        was_blocked = self.safety_blocked
        self.set_safety_block(bool(blocking_rule_ids))
        if was_blocked and not self.safety_blocked:
            self.publish_execution_result(
                "CLEARED",
                "SAFETY_MONITOR",
                "No active safety rule is blocking motion.",
            )

    def set_safety_block(self, blocked):
        """Publish the safety state only when it changes."""
        blocked = bool(blocked)
        if blocked == self.safety_blocked:
            return
        self.safety_blocked = blocked
        self.publish_safety_state(blocked)

    def publish_safety_state(self, blocked):
        """Publish a transient-local motion block for the motion handler."""
        message = Bool()
        message.data = bool(blocked)
        self.safety_stop_publisher.publish(message)

    def publish_execution_result(
        self,
        status,
        intent,
        response,
        data=None,
    ):
        """Publish a safety-management or monitoring event as JSON."""
        result = {
            "handler": "SAFETY_RULE_HANDLER",
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
    """Run the safety rule handler."""
    rclpy.init(args=args)
    node = None
    try:
        node = SafetyRuleHandler()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.publish_safety_state(False)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
