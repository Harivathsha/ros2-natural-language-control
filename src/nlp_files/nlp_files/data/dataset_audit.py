"""Audit the 4,500-row corpus before training or evaluation."""

import argparse
import json
from math import isclose
from numbers import Real
from pathlib import Path

import pandas as pd

from nlp_files.config import (
    DATASET_PATH,
    DECISION_LABELS,
    EXPECTED_ROWS_BY_SPLIT,
    EXPECTED_TOTAL_ROWS,
    INTENT_LABELS,
    SHEET_NAME,
)


REQUIRED_COLUMNS = [
    "example_id",
    "paraphrase_group_id",
    "meaning_id",
    "split",
    "capability",
    "utterance",
    "variation_type",
    "source_type",
    "review_status",
    "intent",
    "expected_decision",
    "sensor",
    "diagnostic_property",
    "action",
    "direction",
    "distance_m",
    "speed_mps",
    "angle_deg",
    "condition_property",
    "operator",
    "threshold_value",
    "threshold_unit",
    "connector",
    "rule_id",
    "missing_slots",
    "semantic_frame_json",
    "validation_reason",
    "expected_system_behavior",
]

NON_NULL_COLUMNS = [
    "example_id",
    "paraphrase_group_id",
    "meaning_id",
    "split",
    "capability",
    "utterance",
    "variation_type",
    "source_type",
    "review_status",
    "intent",
    "expected_decision",
    "semantic_frame_json",
    "validation_reason",
    "expected_system_behavior",
]

CAPABILITY_BY_INTENT = {
    "CHECK_SENSOR_HEALTH": "SENSOR_DIAGNOSIS",
    "CHECK_SENSOR_RATE": "SENSOR_DIAGNOSIS",
    "CHECK_SENSOR_FRESHNESS": "SENSOR_DIAGNOSIS",
    "CHECK_NEAREST_OBSTACLE": "SENSOR_DIAGNOSIS",
    "CHECK_SENSOR_DATA_VALIDITY": "SENSOR_DIAGNOSIS",
    "LIST_UNRESPONSIVE_SENSORS": "SENSOR_DIAGNOSIS",
    "GET_CAMERA_RESOLUTION": "SENSOR_DIAGNOSIS",
    "CHECK_ALL_SENSORS": "SENSOR_DIAGNOSIS",
    "MOVE_DISTANCE": "MOTION_COMMAND",
    "TURN_ANGLE": "MOTION_COMMAND",
    "STOP": "MOTION_COMMAND",
    "CREATE_SAFETY_RULE": "CONDITIONAL_SAFETY",
    "LIST_SAFETY_RULES": "CONDITIONAL_SAFETY",
    "REMOVE_SAFETY_RULE": "CONDITIONAL_SAFETY",
    "CLEAR_SAFETY_RULES": "CONDITIONAL_SAFETY",
    "UNKNOWN": None,
}

SLOT_COLUMNS = [
    "sensor",
    "diagnostic_property",
    "action",
    "direction",
    "distance_m",
    "speed_mps",
    "angle_deg",
    "condition_property",
    "operator",
    "threshold_value",
    "threshold_unit",
    "connector",
    "rule_id",
]


def values_match(expected, predicted):
    """Compare spreadsheet and JSON slot values safely."""
    expected_missing = expected is None or bool(pd.isna(expected))
    predicted_missing = predicted is None
    if expected_missing or predicted_missing:
        return expected_missing and predicted_missing
    if (
        isinstance(expected, Real)
        and not isinstance(expected, bool)
        and isinstance(predicted, Real)
        and not isinstance(predicted, bool)
    ):
        return isclose(float(expected), float(predicted), abs_tol=1e-9)
    return expected == predicted


def load_dataset(file_path, sheet_name):
    """Load an Excel or CSV corpus."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Dataset must be an XLSX, XLS, or CSV file.")


def validate_structure(data):
    """Raise a clear error for structural corpus problems."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")
    if len(data) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ROWS} rows, found {len(data)}."
        )
    if data[NON_NULL_COLUMNS].isna().any().any():
        bad_columns = data[NON_NULL_COLUMNS].isna().sum()
        bad_columns = bad_columns[bad_columns > 0].to_dict()
        raise ValueError(f"Required values are missing: {bad_columns}")
    if data["example_id"].duplicated().any():
        raise ValueError("Duplicate example IDs found.")
    if data["utterance"].duplicated().any():
        raise ValueError("Duplicate utterances found.")

    unknown_intents = set(data["intent"]) - set(INTENT_LABELS)
    if unknown_intents:
        raise ValueError(f"Unsupported intent labels: {sorted(unknown_intents)}")
    unknown_decisions = set(data["expected_decision"]) - set(DECISION_LABELS)
    if unknown_decisions:
        raise ValueError(
            f"Unsupported decision labels: {sorted(unknown_decisions)}"
        )


def validate_splits_and_groups(data):
    """Verify predefined split sizes and prevent paraphrase leakage."""
    for split, expected_rows in EXPECTED_ROWS_BY_SPLIT.items():
        actual_rows = int(data["split"].eq(split).sum())
        if actual_rows != expected_rows:
            raise ValueError(
                f"Split {split!r}: expected {expected_rows}, found {actual_rows}."
            )

    unexpected_splits = set(data["split"]) - set(EXPECTED_ROWS_BY_SPLIT)
    if unexpected_splits:
        raise ValueError(f"Unexpected splits: {sorted(unexpected_splits)}")

    group_sizes = data.groupby("paraphrase_group_id").size()
    invalid_sizes = group_sizes[group_sizes != 5]
    if not invalid_sizes.empty:
        raise ValueError(
            f"{len(invalid_sizes)} paraphrase groups do not contain five rows."
        )

    splits_per_group = data.groupby("paraphrase_group_id")["split"].nunique()
    leaking_groups = splits_per_group[splits_per_group != 1]
    if not leaking_groups.empty:
        raise ValueError(
            f"{len(leaking_groups)} paraphrase groups cross data splits."
        )


def validate_semantic_json(data):
    """Verify stored semantic-frame JSON and its top-level labels."""
    invalid_rows = []
    for _, row in data.iterrows():
        try:
            frame = json.loads(row["semantic_frame_json"])
        except (TypeError, json.JSONDecodeError):
            invalid_rows.append(row["example_id"])
            continue

        expected_capability = CAPABILITY_BY_INTENT[row["intent"]]
        parameters = frame.get("parameters", {})
        if frame.get("intent") != row["intent"]:
            invalid_rows.append(row["example_id"])
        elif (
            expected_capability is not None
            and frame.get("capability") != expected_capability
        ):
            invalid_rows.append(row["example_id"])
        elif not isinstance(parameters, dict):
            invalid_rows.append(row["example_id"])
        elif any(
            not values_match(row[slot_name], parameters.get(slot_name))
            for slot_name in SLOT_COLUMNS
        ):
            invalid_rows.append(row["example_id"])

    if invalid_rows:
        raise ValueError(
            "Invalid semantic_frame_json values for examples: "
            f"{invalid_rows[:20]}"
        )


def print_summary(data):
    """Print distributions needed before model training."""
    print("Dataset rows:", len(data))
    print("Dataset columns:", len(data.columns))
    print("Duplicate example IDs:", int(data["example_id"].duplicated().sum()))
    print("Duplicate utterances:", int(data["utterance"].duplicated().sum()))

    for column in (
        "capability",
        "intent",
        "expected_decision",
        "split",
        "variation_type",
    ):
        print(f"\nCounts for {column}:")
        print(data[column].value_counts(dropna=False).to_string())

    print("\nParaphrase groups:", data["paraphrase_group_id"].nunique())
    print("All structural checks: PASS")


def main():
    """Run the complete corpus audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", default=str(DATASET_PATH))
    parser.add_argument("--sheet", default=SHEET_NAME)
    arguments = parser.parse_args()

    data = load_dataset(arguments.file_path, arguments.sheet)
    validate_structure(data)
    validate_splits_and_groups(data)
    validate_semantic_json(data)
    print_summary(data)


if __name__ == "__main__":
    main()
