"""Shared configuration for the complete ROS 2 NLP package."""

import os
from pathlib import Path


INTENT_LABELS = (
    "CHECK_SENSOR_HEALTH",
    "CHECK_SENSOR_RATE",
    "CHECK_SENSOR_FRESHNESS",
    "CHECK_NEAREST_OBSTACLE",
    "CHECK_SENSOR_DATA_VALIDITY",
    "LIST_UNRESPONSIVE_SENSORS",
    "GET_CAMERA_RESOLUTION",
    "CHECK_ALL_SENSORS",
    "MOVE_DISTANCE",
    "TURN_ANGLE",
    "STOP",
    "CREATE_SAFETY_RULE",
    "LIST_SAFETY_RULES",
    "REMOVE_SAFETY_RULE",
    "CLEAR_SAFETY_RULES",
    "UNKNOWN",
)

ALLOWED_INTENTS = frozenset(INTENT_LABELS)
DECISION_LABELS = ("VALID", "NEEDS_CLARIFICATION", "REJECT")


def _resolve_project_root():
    """Locate the workspace root while allowing an environment override."""
    override = os.environ.get("NLP_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    current_directory = Path.cwd().resolve()
    if (current_directory / "dataset_phase2_9500.xlsx").is_file():
        return current_directory

    conventional_root = (
        Path.home() / "ai_envs" / "project" / "nlp_project"
    )
    if conventional_root.is_dir():
        return conventional_root

    return current_directory


PROJECT_ROOT = _resolve_project_root()
DATASET_PATH = PROJECT_ROOT / "dataset_4500.xlsx"
SHEET_NAME = "Corpus_4500"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "semantic_phase2_typing"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "corpus_4500"

EXPECTED_ROWS_BY_SPLIT = {
    "train": 3150,
    "validation": 675,
    "test": 675,
}
EXPECTED_TOTAL_ROWS = 4500

MAX_DISTANCE_M = 2.0
MAX_SPEED_MPS = 0.3
MAX_ANGLE_DEG = 180.0

MIN_CONFIDENCE = float(os.environ.get("NLP_MIN_CONFIDENCE", "0.25"))

COMMAND_TOPIC = "/nlp/text_command"
RESULT_TOPIC = "/nlp/result"
RESPONSE_TIMEOUT_S = 8.0

ROUTE_BY_CAPABILITY = {
    "NAVIGATION_COMMAND": "NAVIGATION_HANDLER",
    "SENSOR_DIAGNOSIS": "DIAGNOSIS_HANDLER",
    "MOTION_COMMAND": "MOTION_HANDLER",
    "CONDITIONAL_SAFETY": "SAFETY_RULE_HANDLER",
}
