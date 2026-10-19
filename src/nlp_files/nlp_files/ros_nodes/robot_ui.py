#!/usr/bin/env python3
"""A small desktop UI for the working ROS command client. Run in terminal 3."""

import json
import math
import shutil
import subprocess
import tempfile
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from threading import Thread
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import rclpy
import yaml
from PIL import Image as PILImage, ImageTk
from rcl_interfaces.srv import GetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image

from nlp_files.config import COMMAND_TOPIC, PROJECT_ROOT
from nlp_files.ros_nodes.ros_command_client import ROSCommandClient


def camera_image(message):
    """Convert the camera's raw pixels, respecting encoding and row padding."""
    formats = {
        "rgb8": ("RGB", "RGB", 3), "bgr8": ("RGB", "BGR", 3),
        "rgba8": ("RGBA", "RGBA", 4), "bgra8": ("RGBA", "BGRA", 4),
        "mono8": ("L", "L", 1), "8uc1": ("L", "L", 1),
    }
    encoding = message.encoding.lower()
    if encoding not in formats:
        raise ValueError(f"Camera format {message.encoding} is not supported.")
    mode, raw, channels = formats[encoding]
    width, height, stride = int(message.width), int(message.height), int(message.step)
    if width < 1 or height < 1 or stride < width * channels:
        raise ValueError("The camera frame has invalid dimensions.")
    if len(message.data) < stride * height:
        raise ValueError("The camera frame is incomplete.")
    return PILImage.frombytes(mode, (width, height), bytes(message.data),
                             "raw", raw, stride, 1).convert("RGB")


def parameter_text(parameters):
    units = {"distance_m": "m", "speed_mps": "m/s", "angle_deg": "deg"}
    parts = []
    for key, value in parameters.items():
        if key in {"action", "diagnostic_property"}:
            continue
        if key in units:
            label = {"distance_m": "Distance", "speed_mps": "Speed", "angle_deg": "Angle"}[key]
            parts.append(f"{label}: {value} {units[key]}")
        else:
            label = key.replace("_", " ").capitalize()
            text = str(value).replace("_", " ").lower() if isinstance(value, str) else str(value)
            parts.append(f"{label}: {text}")
    return "   ·   ".join(parts)


def export_environment(project, locations_file):
    """Save the current ROS map together with an unchanged copy of its locations."""
    locations_file = Path(locations_file)
    original = locations_file.read_bytes()
    document = json.loads(original)
    locations = document.get("locations", {})
    if document.get("frame_id") != "map" or set(locations) != {"HOME", "LABORATORY"}:
        raise ValueError("Record HOME and LAB before saving the map.")
    for pose in locations.values():
        if (not isinstance(pose, dict) or set(pose) != {"x", "y", "yaw"}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose.values())):
            raise ValueError("The saved location coordinates are invalid.")
    artifacts = Path(project) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    target = artifacts / "saved_environment"
    backup = artifacts / "saved_environment.previous"
    if not target.exists() and backup.is_dir():
        backup.rename(target)
    with tempfile.TemporaryDirectory(prefix=".saving_map_", dir=artifacts) as temporary:
        stage = Path(temporary)
        result = subprocess.run([
            "ros2", "run", "nav2_map_server", "map_saver_cli", "-t", "/map",
            "-f", str(stage / "map"), "--fmt", "pgm", "--mode", "trinary",
            "--ros-args", "-p", "use_sim_time:=true",
            "-p", "map_subscribe_transient_local:=true", "-p", "save_map_timeout:=8.0",
        ], capture_output=True, text=True, timeout=25)
        if result.returncode:
            print(result.stdout, result.stderr, flush=True)
            raise RuntimeError("Map saving failed. Check terminal 3 for the map saver message.")
        map_file, image_file = stage / "map.yaml", stage / "map.pgm"
        if not map_file.is_file() or not image_file.is_file() or image_file.stat().st_size == 0:
            raise RuntimeError("The map saver did not produce a complete map.")
        metadata = yaml.safe_load(map_file.read_text())
        metadata["image"] = "map.pgm"  # Keep the image reference valid after moving the folder.
        map_file.write_text(yaml.safe_dump(metadata))
        if locations_file.read_bytes() != original:
            raise RuntimeError("Locations changed while saving. Stop the robot and save the map again.")
        (stage / "named_locations.json").write_bytes(original)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except Exception:
            if not target.exists() and backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return target


class UIClient(ROSCommandClient):
    """Reuse the tested ROS client; put events in a queue for the window."""

    def __init__(self):
        self.ui_events = deque(maxlen=100)
        self.ui_image = None
        self.ui_image_time = 0.0
        self.ui_image_number = 0
        self.ui_save_name = None
        self.ui_save_started = 0.0
        self.ui_map_busy = False
        self.ui_map_future = None
        self.ui_map_started = 0.0
        super().__init__()
        self.declare_parameter("camera_topic", "/camera/image")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.ui_camera_sub = self.create_subscription(
            Image, self.get_parameter("camera_topic").value, self.on_camera, qos,
        )
        self.ui_nav_parameters = self.create_client(GetParameters, "/navigation_handler/get_parameters")

    def on_camera(self, message):
        self.ui_image = message
        self.ui_image_time = time.monotonic()
        self.ui_image_number += 1

    def nlp_connected(self):
        # The motion handler also listens here; wait for the actual model node.
        return any(info.node_name == "nlp_ros_node"
                   for info in self.get_subscriptions_info_by_topic(COMMAND_TOPIC))

    def on_result(self, message):
        try:
            result = json.loads(message.data)
            if self.pending is not None and result.get("text") == self.pending:
                self.ui_events.append(("nlp", result))
        except (ValueError, AttributeError, TypeError):
            self.ui_events.append(("notice", "The NLP response could not be read."))
        super().on_result(message)

    def on_execution(self, message):
        super().on_execution(message)
        try:
            result = json.loads(message.data)
            if isinstance(result, dict):
                self.ui_events.append(("execution", result))
        except (ValueError, TypeError):
            pass

    def request_save(self, name):
        if self.pending is not None or self.active_handlers or self.ui_save_name or self.ui_map_busy:
            self.ui_events.append(("notice", "Stop the robot and wait for completion before saving a location."))
        elif not self.location_clients[name].service_is_ready():
            self.ui_events.append(("notice", "Navigation is not connected. Check terminal 2."))
        else:
            self.ui_save_name = name
            self.ui_save_started = time.monotonic()
            super().save(name)

    def request_map_save(self):
        if self.pending is not None or self.active_handlers or self.ui_save_name or self.ui_map_busy:
            self.ui_events.append(("notice", "Stop the robot and wait before saving the map."))
            return
        if not self.ui_nav_parameters.service_is_ready():
            self.ui_events.append(("notice", "Navigation is not connected. Keep terminal 2 running."))
            return
        self.ui_map_busy = True
        self.ui_map_started = time.monotonic()
        request = GetParameters.Request()
        request.names = ["locations_file"]
        self.ui_events.append(("notice", "Saving the map and both locations. Please keep this window open."))
        self.ui_map_future = self.ui_nav_parameters.call_async(request)
        self.ui_map_future.add_done_callback(self.map_locations_received)

    def map_locations_received(self, future):
        if not self.ui_map_busy:
            return
        self.ui_map_future = None
        try:
            filename = future.result().values[0].string_value
            if not filename:
                raise ValueError("Navigation did not report a locations file.")
            Thread(target=self.save_map_worker, args=(filename,), daemon=True).start()
        except Exception as error:
            self.ui_map_busy = False
            self.ui_events.append(("notice", f"Could not save the map: {error}"))

    def save_map_worker(self, filename):
        try:
            target = export_environment(PROJECT_ROOT, filename)
            print(f"Saved map and locations: {target}", flush=True)
            self.ui_events.append(("saved", "Map + HOME/LAB saved. They will load on the next start."))
        except Exception as error:
            self.ui_events.append(("notice", f"Could not save the map: {error}"))
        finally:
            self.ui_map_busy = False

    def saved(self, future):
        name = "HOME" if self.ui_save_name == "home" else "LAB"
        self.ui_save_name = None
        try:
            result = future.result()
            text = f"{name} saved at the robot's current position." if result.success else result.message
            self.ui_events.append(("saved" if result.success else "notice", text))
        except Exception as error:
            self.ui_events.append(("notice", f"Could not save {name}: {error}"))

    def poll(self):
        waiting = self.pending is not None
        if waiting and not self.sent and not self.nlp_connected() and time.monotonic() <= self.deadline:
            return
        super().poll()
        if waiting and self.pending is None:
            self.ui_events.append(("timeout", "No NLP reply after 30 seconds. Check terminal 2."))
        if self.ui_save_name and time.monotonic() - self.ui_save_started > 6.0:
            self.ui_save_started = float("inf")
            self.ui_events.append(("notice", "Still waiting for navigation to save the position."))
        if self.ui_map_future is not None and time.monotonic() - self.ui_map_started > 6.0:
            self.ui_map_busy = False
            self.ui_map_future.cancel()
            self.ui_map_future = None
            self.ui_events.append(("notice", "Navigation did not reply. Check terminal 2 and try Save map again."))


class RobotWindow:
    ink = "#202a38"
    muted = "#718096"
    line = "#e5e9ef"
    blue = "#2563eb"

    def __init__(self, root, node):
        self.root, self.node = root, node
        self.closed = False
        self.payload = {}
        self.current_row = None
        self.stage = "Ready"
        self.command_started = 0.0
        self.camera_number = -1
        self.camera_size = None
        self.camera_photo = None
        root.title("Robot control")
        root.geometry("1120x800")
        root.minsize(980, 760)
        root.configure(bg="white")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.configure_style()
        self.build_window()
        root.bind("<Escape>", lambda event: self.stop())
        root.after(30, self.tick)
        root.after(100, self.draw_camera)

    def configure_style(self):
        self.root.option_add("*Font", ("DejaVu Sans", 10))
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", background="white", foreground=self.ink,
                        bordercolor=self.line, lightcolor="white", darkcolor="white",
                        padding=(14, 10), focuscolor="white")
        style.map("TButton", background=[("active", "#f3f6fa")],
                  foreground=[("disabled", "#a3adba")])
        style.configure("Send.TButton", background=self.blue, foreground="white",
                        bordercolor=self.blue, lightcolor=self.blue, darkcolor=self.blue)
        style.map("Send.TButton", background=[("disabled", "#dce5f5"), ("active", "#1d4ed8")],
                  foreground=[("disabled", "#8b9ab4")])
        style.configure("Stop.TButton", foreground="#be3542", bordercolor="#edcbd0")
        style.configure("Treeview", background="white", fieldbackground="white",
                        foreground=self.ink, rowheight=30, borderwidth=0)
        style.configure("Treeview.Heading", background="#f6f8fa", foreground=self.muted,
                        font=("DejaVu Sans", 9), relief="flat")

    def label(self, parent, text="", size=10, bold=False, color=None, **kwargs):
        return tk.Label(parent, text=text, bg="white", fg=color or self.ink,
                        font=("DejaVu Sans", size, "bold" if bold else "normal"),
                        anchor="w", **kwargs)

    def card(self, parent):
        return tk.Frame(parent, bg="white", highlightbackground=self.line,
                        highlightcolor=self.line, highlightthickness=1)

    def build_window(self):
        header = tk.Frame(self.root, bg="white")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(23, 18))
        title = tk.Frame(header, bg="white"); title.pack(side="left")
        self.label(title, "Robot control", size=21, bold=True).pack(anchor="w")
        self.label(title, "Type a command and follow your robot.", color=self.muted).pack(anchor="w", pady=(5, 0))
        self.connection = self.label(header, "●  Waiting for NLP", color=self.muted)
        self.connection.pack(side="right", padx=(16, 0))
        ttk.Button(header, text="■  Stop", style="Stop.TButton", command=self.stop).pack(side="right")

        body = tk.Frame(self.root, bg="white")
        body.grid(row=1, column=0, sticky="nsew", padx=28)
        body.columnconfigure(0, weight=6, uniform="body")
        body.columnconfigure(1, weight=5, uniform="body")
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg="white"); left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        right = tk.Frame(body, bg="white"); right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        compose = self.card(left); compose.pack(fill="x")
        self.label(compose, "Your command", size=12, bold=True).pack(anchor="w", padx=18, pady=(16, 10))
        self.input = tk.Text(compose, height=3, wrap="word", relief="flat", bd=0,
                             bg="#f6f8fb", fg=self.ink, insertbackground=self.blue,
                             padx=12, pady=10, font=("DejaVu Sans", 11), highlightthickness=1,
                             highlightbackground=self.line, highlightcolor=self.blue)
        self.input.pack(fill="x", padx=18)
        self.input.bind("<Return>", self.enter)
        actions = tk.Frame(compose, bg="white"); actions.pack(fill="x", padx=18, pady=(12, 16))
        self.label(actions, "Enter to send · Shift+Enter for a new line", size=8, color=self.muted).pack(side="left")
        self.send_button = ttk.Button(actions, text="Send  →", style="Send.TButton", command=self.send)
        self.send_button.pack(side="right")

        result = self.card(left); result.pack(fill="both", expand=True, pady=(16, 0))
        result.columnconfigure(0, weight=1)
        result.rowconfigure(3, weight=1)
        top = tk.Frame(result, bg="white"); top.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 14))
        self.label(top, "Result", size=12, bold=True).pack(side="left")
        self.badge = self.label(top, "Ready", size=9, color=self.muted, padx=9, pady=4)
        self.badge.pack(side="right")
        self.result_title = self.label(result, "Ready when you are", size=16, bold=True)
        self.result_title.grid(row=1, column=0, sticky="ew", padx=18)
        self.summary = self.label(result, "", size=9, color=self.muted, wraplength=430, justify="left")
        self.summary.grid(row=2, column=0, sticky="ew", padx=18, pady=(9, 0))
        self.response = self.label(result, "Try a movement, a sensor check or a trip home.",
                                   size=11, wraplength=430, justify="left")
        self.response.configure(anchor="nw")
        self.response.grid(row=3, column=0, sticky="nsew", padx=18, pady=(18, 16))
        bottom = tk.Frame(result, bg="white"); bottom.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 14))
        self.label(bottom, "Wait for movement to finish, or use Stop.", size=8, color=self.muted).pack(side="left")
        self.details_button = ttk.Button(bottom, text="Details", command=self.details, state="disabled")
        self.details_button.pack(side="right")

        camera = self.card(right); camera.grid(row=0, column=0, sticky="nsew")
        camera.columnconfigure(0, weight=1)
        camera.rowconfigure(1, weight=1)
        camera_header = tk.Frame(camera, bg="white")
        camera_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 12))
        self.label(camera_header, "Camera", size=12, bold=True).pack(side="left")
        self.camera_status = self.label(camera_header, "Waiting", size=9, color=self.muted)
        self.camera_status.pack(side="right")
        self.camera_canvas = tk.Canvas(camera, bg="#f4f6f8", bd=0, highlightthickness=0, height=220)
        self.camera_canvas.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.camera_caption = self.label(camera, "Robot's view", size=8, color=self.muted)
        self.camera_caption.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

        locations = self.card(right); locations.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        self.label(locations, "Locations", size=12, bold=True).pack(anchor="w", padx=16, pady=(14, 6))
        self.label(locations, "Save the current position while the robot is stopped.",
                   size=8, color=self.muted).pack(anchor="w", padx=16)
        buttons = tk.Frame(locations, bg="white"); buttons.pack(fill="x", padx=16, pady=(12, 10))
        buttons.columnconfigure((0, 1, 2), weight=1, uniform="save")
        self.home_button = ttk.Button(buttons, text="Save HOME", command=lambda: self.node.request_save("home"))
        self.home_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.lab_button = ttk.Button(buttons, text="Save LAB", command=lambda: self.node.request_save("lab"))
        self.lab_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.map_button = ttk.Button(buttons, text="Save map", command=self.node.request_map_save)
        self.map_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))
        self.location_note = self.label(locations, "Save both locations, then save the map once for future starts.",
                                        size=8, color=self.muted, wraplength=420, justify="left")
        self.location_note.pack(fill="x", padx=16, pady=(0, 13))

        history_box = self.card(self.root)
        history_box.grid(row=2, column=0, sticky="ew", padx=28, pady=(18, 20))
        self.label(history_box, "Recent commands", size=10, bold=True).pack(anchor="w", padx=16, pady=(12, 9))
        self.history = ttk.Treeview(history_box, columns=("time", "command", "status"), show="headings", height=3)
        for name, title, width in (("time", "Time", 80), ("command", "Command", 690), ("status", "Result", 145)):
            self.history.heading(name, text=title, anchor="w")
            self.history.column(name, width=width, stretch=name == "command", anchor="w")
        self.history.pack(fill="x", padx=16, pady=(0, 12))
        self.input.focus_set()

    def enter(self, event):
        if event.state & 1:
            return None
        self.send()
        return "break"

    def set_stage(self, text):
        self.stage = text
        color, background = {
            "Completed": ("#237b58", "#edf7f1"), "Running": (self.blue, "#edf3ff"),
            "Understanding": (self.blue, "#edf3ff"), "Accepted": (self.blue, "#edf3ff"),
            "Rejected": ("#a56516", "#fff5e7"), "Needs detail": ("#a56516", "#fff5e7"),
            "Failed": ("#b93543", "#fff0f2"), "Stopped": (self.muted, "#f2f4f7"),
        }.get(text, (self.muted, "#f2f4f7"))
        self.badge.configure(text=text, fg=color, bg=background)
        if self.current_row and self.history.exists(self.current_row):
            self.history.set(self.current_row, "status", text)

    def new_history(self, command):
        self.current_row = self.history.insert("", 0, values=(time.strftime("%H:%M"), command, ""))
        for row in self.history.get_children()[30:]:
            self.history.delete(row)

    def send(self):
        text = self.input.get("1.0", "end").strip()
        if not text:
            self.input.focus_set()
            return
        if self.node.pending is not None or self.node.active_handlers or self.node.ui_save_name or self.node.ui_map_busy:
            return
        if not self.node.nlp_connected():
            self.response.configure(text="The command node is not connected yet. Check terminal 2.")
            return
        self.payload = {"command": text}
        self.new_history(text)
        self.set_stage("Understanding")
        self.result_title.configure(text="Understanding your command")
        self.summary.configure(text="")
        self.response.configure(text=text)
        self.details_button.configure(state="normal")
        self.node.submit(text)
        self.command_started = time.monotonic()
        self.input.delete("1.0", "end")

    def stop(self):
        self.node.stop()
        self.payload = {"operator_control": "STOP"}
        self.new_history("Stop (button)")
        self.set_stage("Stopping")
        self.result_title.configure(text="Stop requested")
        self.summary.configure(text="")
        self.response.configure(text="Waiting for the motion and navigation handlers to confirm.")
        self.details_button.configure(state="normal")

    def receive(self, kind, data):
        if kind == "nlp":
            self.payload["nlp"] = data
            frame = data.get("semantic_frame", {})
            intent = frame.get("intent", "UNKNOWN").replace("_", " ").capitalize()
            decision = data.get("validation", {}).get("decision")
            self.result_title.configure(text=intent)
            self.summary.configure(text=parameter_text(frame.get("parameters", {})))
            if decision != "VALID":
                self.set_stage("Needs detail" if decision == "NEEDS_CLARIFICATION" else "Rejected")
                self.response.configure(text=data.get("response", "Command was not executed."))
            elif self.stage == "Understanding":
                self.set_stage("Accepted")
                self.response.configure(text="Command understood. Waiting for the robot.")
        elif kind == "execution":
            self.payload["execution"] = data
            status = data.get("status", "")
            if self.payload.get("operator_control") == "STOP":
                if status == "FAILED":
                    self.set_stage("Failed")
                elif not self.node.active_handlers and status in {"COMPLETED", "INTERRUPTED", "IDLE"}:
                    self.set_stage("Stopped")
                    self.result_title.configure(text="Robot stopped")
                else:
                    self.set_stage("Stopping")
            elif status == "STARTED":
                self.set_stage("Running")
            elif status == "COMPLETED":
                self.set_stage("Stopped" if data.get("intent") == "STOP" else "Completed")
            elif status in {"FAILED", "REJECTED", "INTERRUPTED"}:
                self.set_stage({"FAILED": "Failed", "REJECTED": "Rejected", "INTERRUPTED": "Stopped"}[status])
            self.response.configure(text=data.get("response", ""))
        elif kind in {"saved", "notice"}:
            self.location_note.configure(text=data, fg="#237b58" if kind == "saved" else self.muted)
        elif kind == "timeout":
            self.set_stage("Failed")
            self.response.configure(text=data)
        self.details_button.configure(state="normal" if self.payload else "disabled")

    def tick(self):
        if self.closed:
            return
        if not rclpy.ok():
            self.close()
            return
        for _ in range(8):
            rclpy.spin_once(self.node, timeout_sec=0)
        self.node.poll()
        while self.node.ui_events:
            self.receive(*self.node.ui_events.popleft())
        if self.stage == "Accepted" and time.monotonic() - self.command_started > 30.0:
            self.set_stage("Failed")
            self.response.configure(text="No execution reply received. Check the handlers in terminal 2.")
        connected = self.node.nlp_connected()
        self.connection.configure(text="●  NLP connected" if connected else "●  Waiting for NLP",
                                  fg="#237b58" if connected else self.muted)
        busy = (self.node.pending is not None or bool(self.node.active_handlers)
                or self.node.ui_save_name is not None or self.node.ui_map_busy
                or self.stage in {"Understanding", "Accepted", "Stopping"})
        self.send_button.configure(state="normal" if connected and not busy else "disabled")
        for button in (self.home_button, self.lab_button, self.map_button):
            button.configure(state="disabled" if busy else "normal")
        self.root.after(35, self.tick)

    def draw_camera(self):
        if self.closed:
            return
        canvas = self.camera_canvas
        width, height = max(10, canvas.winfo_width()), max(10, canvas.winfo_height())
        age = time.monotonic() - self.node.ui_image_time
        self.camera_status.configure(text="Live" if age < 2.0 else "Waiting" if not self.node.ui_image else "Paused",
                                     fg="#237b58" if age < 2.0 else self.muted)
        if self.node.ui_image is None:
            canvas.delete("all")
            canvas.create_text(width/2, height/2, text="Waiting for the camera", fill=self.muted,
                               font=("DejaVu Sans", 11))
        elif self.camera_number != self.node.ui_image_number or self.camera_size != (width, height):
            try:
                message = self.node.ui_image
                image = camera_image(message)
                image.thumbnail((width, height), PILImage.Resampling.BILINEAR)
                self.camera_photo = ImageTk.PhotoImage(image, master=self.root)
                canvas.delete("all")
                canvas.create_image(width/2, height/2, image=self.camera_photo)
                self.camera_caption.configure(text=f"Robot's view  ·  {message.width} × {message.height}")
            except (ValueError, OSError, TypeError) as error:
                self.camera_caption.configure(text=str(error))
            self.camera_number, self.camera_size = self.node.ui_image_number, (width, height)
        self.root.after(100, self.draw_camera)

    def details(self):
        popup = tk.Toplevel(self.root)
        popup.title("Command details")
        popup.geometry("700x500")
        text = ScrolledText(popup, bg="white", fg=self.ink, relief="flat", padx=18, pady=18,
                            font=("DejaVu Sans Mono", 10), wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", json.dumps(self.payload, indent=2, ensure_ascii=False))
        text.configure(state="disabled")

    def close(self):
        if self.node.ui_map_busy and rclpy.ok():
            self.location_note.configure(text="Map saving is in progress. Please wait before closing.")
            return
        if not self.closed:
            self.closed = True
            self.root.destroy()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node, root = None, None
    try:
        node = UIClient()
        root = tk.Tk()
        RobotWindow(root, node)
        root.mainloop()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except tk.TclError as error:
        print(f"Could not open the UI: {error}. Run it from your Ubuntu desktop terminal.")
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
                until = time.monotonic() + 0.4
                while rclpy.ok() and time.monotonic() < until:
                    rclpy.spin_once(node, timeout_sec=0.02)
            node.destroy_node()
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
