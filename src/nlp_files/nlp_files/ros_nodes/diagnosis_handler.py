#!/usr/bin/env python3
"""Answer validated NLP diagnostic requests from live ROS sensor topics."""

import json
import math
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import String

from nlp_files.config import RESULT_TOPIC


class DiagnosisHandler(Node):
    """Cache sensor observations and answer the eight diagnosis intents."""

    SENSORS = ("LIDAR", "CAMERA", "IMU")

    def __init__(self):
        super().__init__("diagnosis_handler")

        self.declare_parameter("nlp_result_topic", RESULT_TOPIC)
        self.declare_parameter("lidar_topic", "/scan")
        self.declare_parameter("camera_topic", "/camera/image")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter(
            "execution_result_topic",
            "/nlp/execution_result",
        )
        self.declare_parameter("sensor_timeout_s", 2.0)

        self.sensor_timeout_s = float(
            self.get_parameter("sensor_timeout_s").value
        )
        self.latest_scan = None
        self.latest_image = None
        self.latest_imu = None
        self.nearest_obstacle_m = None
        self.camera_resolution = None

        self.last_received = {sensor: None for sensor in self.SENSORS}
        self.data_valid = {sensor: False for sensor in self.SENSORS}
        self.message_times = {
            sensor: deque(maxlen=30) for sensor in self.SENSORS
        }

        self.execution_result_publisher = self.create_publisher(
            String,
            self.get_parameter("execution_result_topic").value,
            10,
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

        self.get_logger().info(
            "Diagnosis handler ready for LiDAR, camera, and IMU requests."
        )

    def now_seconds(self):
        """Return current ROS time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def record_message_time(self, sensor):
        """Record receipt time for freshness and rate calculations."""
        current_time = self.now_seconds()
        self.last_received[sensor] = current_time
        self.message_times[sensor].append(current_time)

    def scan_callback(self, message):
        """Cache LiDAR validity and nearest finite obstacle distance."""
        self.latest_scan = message
        metadata_valid = (
            math.isfinite(message.range_min)
            and math.isfinite(message.range_max)
            and 0.0 <= message.range_min < message.range_max
        )
        acceptable_samples = [
            value
            for value in message.ranges
            if (
                math.isinf(value)
                and value > 0.0
            )
            or (
                math.isfinite(value)
                and message.range_min <= value <= message.range_max
            )
        ]
        finite_ranges = [
            float(value)
            for value in message.ranges
            if math.isfinite(value)
            and message.range_min <= value <= message.range_max
        ]

        self.data_valid["LIDAR"] = (
            metadata_valid
            and bool(message.ranges)
            and bool(acceptable_samples)
        )
        self.nearest_obstacle_m = (
            min(finite_ranges) if finite_ranges else None
        )
        self.record_message_time("LIDAR")

    def camera_callback(self, message):
        """Cache camera resolution and validate the image payload."""
        self.latest_image = message
        self.camera_resolution = {
            "width": int(message.width),
            "height": int(message.height),
            "encoding": str(message.encoding),
        }
        expected_size = int(message.height) * int(message.step)
        self.data_valid["CAMERA"] = (
            message.width > 0
            and message.height > 0
            and message.step > 0
            and bool(message.encoding)
            and len(message.data) >= expected_size
        )
        self.record_message_time("CAMERA")

    def imu_callback(self, message):
        """Cache the IMU and reject non-finite acceleration or gyro data."""
        self.latest_imu = message
        values = (
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
            message.angular_velocity.x,
            message.angular_velocity.y,
            message.angular_velocity.z,
        )
        self.data_valid["IMU"] = all(
            math.isfinite(value) for value in values
        )
        self.record_message_time("IMU")

    def get_message_age(self, sensor):
        """Return seconds since the last message, or None before first data."""
        last_time = self.last_received.get(sensor)
        if last_time is None:
            return None
        return max(0.0, self.now_seconds() - last_time)

    def get_message_rate(self, sensor):
        """Estimate average receive rate over the recent message window."""
        times = self.message_times.get(sensor)
        if times is None or len(times) < 2:
            return None
        if self.get_message_age(sensor) > self.sensor_timeout_s:
            return 0.0
        duration = times[-1] - times[0]
        if duration <= 0.0:
            return None
        return (len(times) - 1) / duration

    def get_sensor_health(self, sensor):
        """Create a serialisable health snapshot for one sensor."""
        age = self.get_message_age(sensor)
        rate = self.get_message_rate(sensor)
        responsive = age is not None and age <= self.sensor_timeout_s
        valid = self.data_valid.get(sensor, False)

        if not responsive:
            state = "UNRESPONSIVE"
        elif not valid:
            state = "INVALID_DATA"
        else:
            state = "HEALTHY"

        return {
            "sensor": sensor,
            "state": state,
            "responsive": responsive,
            "data_valid": valid,
            "age_s": None if age is None else round(age, 3),
            "rate_hz": None if rate is None else round(rate, 2),
        }

    def nlp_result_callback(self, message):
        """Handle only valid frames routed to the diagnosis handler."""
        try:
            result = json.loads(message.data)
        except (json.JSONDecodeError, TypeError) as error:
            self.get_logger().error(f"Invalid NLP result JSON: {error}")
            return

        if not isinstance(result, dict):
            self.get_logger().error("NLP result must be a JSON object.")
            return
        if result.get("route") != "DIAGNOSIS_HANDLER":
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
                "Diagnostic parameters are malformed.",
            )
            return

        status, response, data = self.handle_diagnosis(intent, parameters)
        self.publish_execution_result(status, intent, response, data)

    def handle_diagnosis(self, intent, parameters):
        """Return status, human response, and structured diagnostic data."""
        sensor = parameters.get("sensor")

        if intent == "CHECK_SENSOR_HEALTH":
            if sensor not in self.SENSORS:
                return "REJECTED", "The requested sensor is unsupported.", {}
            health = self.get_sensor_health(sensor)
            status = "COMPLETED" if health["responsive"] else "UNAVAILABLE"
            response = f"{sensor} state is {health['state']}."
            return status, response, health

        if intent == "CHECK_SENSOR_RATE":
            if sensor not in self.SENSORS:
                return "REJECTED", "The requested sensor is unsupported.", {}
            health = self.get_sensor_health(sensor)
            rate = health["rate_hz"]
            if not health["responsive"]:
                return (
                    "UNAVAILABLE",
                    f"{sensor} is not currently responding.",
                    health,
                )
            if rate is None:
                return (
                    "UNAVAILABLE",
                    f"Not enough {sensor} messages to calculate its rate.",
                    health,
                )
            return (
                "COMPLETED",
                f"{sensor} is publishing at approximately {rate:.2f} Hz.",
                health,
            )

        if intent == "CHECK_SENSOR_FRESHNESS":
            if sensor not in self.SENSORS:
                return "REJECTED", "The requested sensor is unsupported.", {}
            health = self.get_sensor_health(sensor)
            age = health["age_s"]
            if age is None:
                return "UNAVAILABLE", f"No {sensor} message has arrived.", health
            return (
                "COMPLETED",
                f"The latest {sensor} message is {age:.3f} seconds old.",
                health,
            )

        if intent == "CHECK_NEAREST_OBSTACLE":
            health = self.get_sensor_health("LIDAR")
            if not health["responsive"]:
                return "UNAVAILABLE", "LiDAR is not responding.", health
            if not health["data_valid"]:
                return "UNAVAILABLE", "LiDAR data is invalid.", health
            data = dict(health)
            data["nearest_obstacle_m"] = self.nearest_obstacle_m
            if self.nearest_obstacle_m is None:
                return (
                    "COMPLETED",
                    "No finite obstacle was detected within the LiDAR range.",
                    data,
                )
            return (
                "COMPLETED",
                "The nearest obstacle is "
                f"{self.nearest_obstacle_m:.3f} metres away.",
                data,
            )

        if intent == "CHECK_SENSOR_DATA_VALIDITY":
            if sensor not in self.SENSORS:
                return "REJECTED", "The requested sensor is unsupported.", {}
            health = self.get_sensor_health(sensor)
            if self.last_received[sensor] is None:
                return "UNAVAILABLE", f"No {sensor} data has arrived.", health
            validity = "valid" if health["data_valid"] else "invalid"
            return (
                "COMPLETED",
                f"The latest {sensor} data is {validity}.",
                health,
            )

        if intent == "LIST_UNRESPONSIVE_SENSORS":
            health_by_sensor = {
                name: self.get_sensor_health(name) for name in self.SENSORS
            }
            unresponsive = [
                name
                for name, health in health_by_sensor.items()
                if not health["responsive"]
            ]
            response = (
                "All configured sensors are responding."
                if not unresponsive
                else "Unresponsive sensors: " + ", ".join(unresponsive) + "."
            )
            return (
                "COMPLETED",
                response,
                {
                    "unresponsive_sensors": unresponsive,
                    "sensors": health_by_sensor,
                },
            )

        if intent == "GET_CAMERA_RESOLUTION":
            health = self.get_sensor_health("CAMERA")
            if not health["responsive"] or self.camera_resolution is None:
                return "UNAVAILABLE", "Camera data is unavailable.", health
            data = dict(health)
            data["resolution"] = self.camera_resolution
            return (
                "COMPLETED",
                "Camera resolution is "
                f"{self.camera_resolution['width']} x "
                f"{self.camera_resolution['height']} pixels.",
                data,
            )

        if intent == "CHECK_ALL_SENSORS":
            health_by_sensor = {
                name: self.get_sensor_health(name) for name in self.SENSORS
            }
            unhealthy = [
                name
                for name, health in health_by_sensor.items()
                if health["state"] != "HEALTHY"
            ]
            response = (
                "All configured sensors are healthy."
                if not unhealthy
                else "Sensors needing attention: " + ", ".join(unhealthy) + "."
            )
            return "COMPLETED", response, {"sensors": health_by_sensor}

        return (
            "REJECTED",
            f"Unsupported diagnostic intent: {intent}",
            {},
        )

    def publish_execution_result(
        self,
        status,
        intent,
        response,
        data=None,
    ):
        """Publish one diagnostic result as JSON."""
        result = {
            "handler": "DIAGNOSIS_HANDLER",
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
    """Run the diagnosis handler."""
    rclpy.init(args=args)
    node = None
    try:
        node = DiagnosisHandler()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
