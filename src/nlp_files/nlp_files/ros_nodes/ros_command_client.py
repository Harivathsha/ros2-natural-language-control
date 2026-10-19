#!/usr/bin/env python3
"""Terminal 3: interactive ROS commands, live execution results and pose saving."""

import json
import select
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from std_srvs.srv import Trigger

from nlp_files.config import COMMAND_TOPIC, RESULT_TOPIC


class ROSCommandClient(Node):
    def __init__(self):
        super().__init__("ros_command_client")
        self.pending = None
        self.sent = False
        self.deadline = None
        self.active_handlers = set()
        self.commands = self.create_publisher(String, COMMAND_TOPIC, 10)
        self.operator_commands = self.create_publisher(String, RESULT_TOPIC, 10)
        self.results = self.create_subscription(String, RESULT_TOPIC, self.on_result, 10)
        self.execution = self.create_subscription(
            String, "/nlp/execution_result", self.on_execution, 10,
        )
        self.location_clients = {
            "home": self.create_client(Trigger, "/nlp/save_home"),
            "lab": self.create_client(Trigger, "/nlp/save_laboratory"),
        }

    def submit(self, command):
        if self.pending is not None:
            print("Wait for the current NLP response, or use :stop.")
            return
        self.pending, self.sent = command, False
        self.deadline = time.monotonic() + 30.0

    def poll(self):
        if self.pending is None:
            return
        if time.monotonic() > self.deadline:
            print("\nNo NLP response after 30 seconds. Check terminal 2.")
            self.pending = None
        elif not self.sent and self.commands.get_subscription_count():
            self.commands.publish(String(data=self.pending))
            self.sent = True

    def on_result(self, message):
        try:
            result = json.loads(message.data)
            if self.pending is None or result.get("text") != self.pending:
                return
            self.pending = None
            frame = result.get("semantic_frame", {})
            decision = result.get("validation", {}).get("decision", "UNKNOWN")
            print(f"\n[NLP {decision}] {frame.get('intent')} "
                  f"{json.dumps(frame.get('parameters', {}))}")
            if decision != "VALID":
                print(result.get("response", "Command not executed."))
        except (ValueError, AttributeError, TypeError):
            print("\nInvalid NLP response. Check terminal 2.")

    def on_execution(self, message):
        try:
            result = json.loads(message.data)
            handler = result.get("handler")
            if handler in {"MOTION_HANDLER", "NAVIGATION_HANDLER"}:
                if result.get("status") == "STARTED":
                    self.active_handlers.add(handler)
                elif result.get("status") in {"COMPLETED", "INTERRUPTED", "IDLE"}:
                    self.active_handlers.discard(handler)
                elif result.get("status") == "FAILED" and handler == "MOTION_HANDLER":
                    self.active_handlers.discard(handler)
            print(f"\n[{result.get('handler')} {result.get('status')}] "
                  f"{result.get('response')}")
            data = result.get("data")
            if data:
                print(json.dumps(data, indent=2, ensure_ascii=False))
        except (ValueError, AttributeError, TypeError):
            print(message.data)

    def save(self, location):
        if self.pending is not None or self.active_handlers:
            print("Finish the movement or use :stop and wait for cancellation before saving.")
            return
        client = self.location_clients[location]
        if not client.service_is_ready():
            print("Navigation handler is not ready. Check terminal 2.")
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(self.saved)

    @staticmethod
    def saved(future):
        try:
            result = future.result()
            print(f"\n[{'SAVED' if result.success else 'NOT SAVED'}] {result.message}")
        except Exception as error:
            print(f"\nCould not save location: {error}")

    def stop(self):
        # Operator button, not a language/spelling rule: send a canonical STOP.
        self.pending = None
        self.operator_commands.publish(String(data=json.dumps({
            "route": "MOTION_HANDLER", "validation": {"decision": "VALID"},
            "semantic_frame": {"schema_version": "1.0", "capability": "MOTION_COMMAND",
                               "intent": "STOP", "parameters": {}},
        })))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ROSCommandClient()
    print("Type robot commands; wait for COMPLETED before the next movement.")
    print("Setup controls: :save home | :save lab (robot stopped); :stop; quit")
    print("Saved maps reuse HOME/LAB. Record positions again when making a new map.")
    print("Command> ", end="", flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.poll()
            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            command = line.strip()
            if command.lower() in {"quit", "exit"}:
                break
            if command == ":stop":
                node.stop()
            elif command == ":save home":
                node.save("home")
            elif command == ":save lab":
                node.save("lab")
            elif command:
                node.submit(command)
            print("Command> ", end="", flush=True)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
            until = time.monotonic() + 0.5
            while rclpy.ok() and time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
