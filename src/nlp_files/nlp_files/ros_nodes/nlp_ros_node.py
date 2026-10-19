"""Connect the standalone NLP pipeline to ROS 2 String topics."""

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from nlp_files.config import (
    COMMAND_TOPIC,
    MIN_CONFIDENCE,
    MODEL_PATH,
    RESULT_TOPIC,
)
from nlp_files.core.nlp_pipeline import NLPPipeline


class NLPRosNode(Node):
    """Receive text commands and publish complete NLP results as JSON."""

    def __init__(self):
        super().__init__("nlp_ros_node")

        self.declare_parameter("model_path", str(MODEL_PATH))
        self.declare_parameter("min_confidence", MIN_CONFIDENCE)
        self.declare_parameter("command_topic", COMMAND_TOPIC)
        self.declare_parameter("result_topic", RESULT_TOPIC)

        model_path = self.get_parameter("model_path").value
        min_confidence = self.get_parameter("min_confidence").value
        self.command_topic = self.get_parameter("command_topic").value
        self.result_topic = self.get_parameter("result_topic").value

        self.pipeline = NLPPipeline(
            model_path=model_path,
            min_confidence=min_confidence,
        )
        self.result_publisher = self.create_publisher(
            String,
            self.result_topic,
            10,
        )
        self.command_subscription = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )

        self.get_logger().info(
            f"NLP ROS node ready: {self.command_topic} -> {self.result_topic}"
        )

    @staticmethod
    def make_error_result(text):
        """Create a fail-safe rejection after an unexpected internal error."""
        return {
            "text": text,
            "raw_intent": "UNKNOWN",
            "selected_intent": "UNKNOWN",
            "intent_source": "internal_error",
            "confidence": 0.0,
            "semantic_frame": {
                "schema_version": "1.0",
                "capability": "ANY",
                "intent": "UNKNOWN",
                "parameters": {},
            },
            "validation": {
                "decision": "REJECT",
                "reason": "The NLP pipeline could not process this request.",
                "missing_slots": [],
            },
            "route": "NO_EXECUTION",
            "response": (
                "Command rejected because the NLP pipeline encountered "
                "an internal error."
            ),
        }

    def command_callback(self, message):
        """Process one command and always publish one JSON response."""
        text = message.data.strip()
        try:
            result = self.pipeline.process(text)
        except Exception as error:
            self.get_logger().error(
                f"NLP processing failed: {type(error).__name__}: {error}"
            )
            result = self.make_error_result(text)

        outgoing_message = String()
        outgoing_message.data = json.dumps(result, ensure_ascii=False)
        self.result_publisher.publish(outgoing_message)

        validation = result["validation"]
        self.get_logger().info(
            f"intent={result['selected_intent']}, "
            f"decision={validation['decision']}, "
            f"route={result['route']}"
        )


def main(args=None):
    """Start the ROS node and clean it up safely."""
    rclpy.init(args=args)
    node = None
    try:
        node = NLPRosNode()
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
