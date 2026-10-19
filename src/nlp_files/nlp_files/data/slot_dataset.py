"""Load the Phase 2 corpus, validate BIO spans and prepare slot-model inputs."""

import argparse
import json
from math import isfinite
from pathlib import Path

import pandas as pd

from nlp_files.config import DECISION_LABELS, PROJECT_ROOT
from nlp_files.config import INTENT_LABELS as PHASE1_INTENT_LABELS
from nlp_files.data import dataset_audit as phase1_audit


# Phase 2 settings stay here so the frozen Phase 1 configuration keeps working.
DATASET_PATH = PROJECT_ROOT / "dataset_phase2_9500.xlsx"
SHEET_NAME = "Corpus_9500"
EXPECTED_TOTAL_ROWS = 9500
EXPECTED_ROWS_BY_SPLIT = {"train": 7300, "validation": 1100, "test": 1100}
CORPUS_LAYOUTS = {
    "Corpus_5000": {"train": 3500, "validation": 750, "test": 750},
    "Corpus_8500": {"train": 6500, "validation": 1000, "test": 1000},
    SHEET_NAME: EXPECTED_ROWS_BY_SPLIT,
}
INTENT_LABELS = (*PHASE1_INTENT_LABELS, "NAVIGATE_TO_LOCATION")
SUPPORTED_LOCATIONS = ("HOME", "LABORATORY")


JSON_COLUMNS = {
    "tokens": "tokens_json",
    "token_offsets": "token_offsets_json",
    "bio_tags": "bio_tags_json",
    "slot_spans": "slot_spans_json",
}

SLOT_BY_LABEL = {
    "SENSOR": "sensor",
    "DIRECTION": "direction",
    "DISTANCE": "distance_m",
    "SPEED": "speed_mps",
    "ANGLE": "angle_deg",
    "ACTION": "action",
    "CONNECTOR": "connector",
    "CONDITION_PROPERTY": "condition_property",
    "OPERATOR": "operator",
    "THRESHOLD": "threshold_value",
    "RULE_ID": "rule_id",
    "LOCATION": "location",
}

ALLOWED_BIO_TAGS = ["O"]
for label in SLOT_BY_LABEL:
    ALLOWED_BIO_TAGS.append("B-" + label)
    ALLOWED_BIO_TAGS.append("I-" + label)

NUMERIC_LABELS = ["DISTANCE", "SPEED", "ANGLE", "THRESHOLD"]
SOURCE_COLUMNS = phase1_audit.REQUIRED_COLUMNS
SLOT_MODEL_NAME = "distilbert/distilbert-base-uncased"
IGNORE_LABEL_ID = -100
MAX_SEQUENCE_LENGTH = 128


def parse_json_column(value, column_name, example_id):
    """Turn a JSON cell into a Python list."""
    message = f"Example {example_id}, column {column_name}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{message}: expected a nonempty JSON string.")
    try:
        result = json.loads(value)
        # Standard JSON cannot contain NaN or Infinity, even inside a list.
        json.dumps(result, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"{message}: invalid JSON.") from error
    if not isinstance(result, list):
        raise ValueError(f"{message}: expected a JSON list.")
    return result


def row_to_record(row):
    """Copy a spreadsheet row and add its four decoded annotation lists."""
    record = dict(row)  # Works for a dictionary or a pandas Series.
    example_id = record.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError("The row needs a nonempty example_id.")
    for name, column in JSON_COLUMNS.items():
        if column not in record:
            raise ValueError(f"Example {example_id}: missing {column}.")
        record[name] = parse_json_column(record[column], column, example_id)
    return record


def validate_record(record):
    """Check tokens and BIO labels, then check the complete entity spans."""
    message = f"Example {record.get('example_id', '<missing>')}"
    text = record.get("utterance")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{message}: utterance must be a nonempty string.")
    for name in JSON_COLUMNS:
        if not isinstance(record.get(name), list):
            raise ValueError(f"{message}: {name} must be a decoded list.")

    tokens = record["tokens"]
    offsets = record["token_offsets"]
    tags = record["bio_tags"]
    if not tokens or not (len(tokens) == len(offsets) == len(tags)):
        raise ValueError(f"{message}: tokens, offsets and BIO lengths must match.")

    previous_end = 0
    previous_tag = "O"
    bio_spans = []

    for token, offset, tag in zip(tokens, offsets, tags):
        if not isinstance(token, str) or not token.strip():
            raise ValueError(f"{message}: token must be a nonempty string.")
        if not isinstance(offset, list) or len(offset) != 2:
            raise ValueError(f"{message}: offset needs two positions.")
        start, end = offset
        # Exact type checks exclude True and False, which behave like integers.
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"{message}: offset positions must be integers.")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{message}: offset is outside the utterance.")
        if start < previous_end:
            raise ValueError(f"{message}: token overlaps or is out of order.")
        if text[previous_end:start].strip():
            raise ValueError(f"{message}: uncovered non-whitespace before a token.")
        if text[start:end] != token:
            raise ValueError(f"{message}: token does not match its substring.")

        if not isinstance(tag, str) or tag not in ALLOWED_BIO_TAGS:
            raise ValueError(f"{message}: unsupported BIO tag.")
        if tag.startswith("B-"):
            bio_spans.append([tag[2:], start, end])
        elif tag.startswith("I-"):
            label = tag[2:]
            if previous_tag not in ("B-" + label, "I-" + label):
                raise ValueError(f"{message}: invalid BIO continuation.")
            bio_spans[-1][2] = end
        previous_end = end
        previous_tag = tag

    if text[previous_end:].strip():
        raise ValueError(f"{message}: uncovered non-whitespace after the last token.")
    validate_spans(record, bio_spans)
    validate_navigation_record(record)


def validate_spans(record, bio_spans):
    """Compare saved spans with the text, BIO boundaries, and source values."""
    message = f"Example {record['example_id']}"
    text = record["utterance"]
    required = ["slot", "label", "text", "start", "end", "normalized_value"]
    saved_spans = []
    previous_end = 0

    for span in record["slot_spans"]:
        if not isinstance(span, dict):
            raise ValueError(f"{message}: each span must be a dictionary.")
        for field in required:
            if field not in span:
                raise ValueError(f"{message}: span is missing {field}.")
        label = span["label"]
        if not isinstance(label, str) or label not in SLOT_BY_LABEL:
            raise ValueError(f"{message}: unsupported span label.")
        if span["slot"] != SLOT_BY_LABEL[label]:
            raise ValueError(f"{message}: span label and slot name disagree.")

        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"{message}: span positions must be integers.")
        if not 0 <= start < end <= len(text) or start < previous_end:
            raise ValueError(f"{message}: invalid or overlapping span positions.")
        if span["text"] != text[start:end]:
            raise ValueError(f"{message}: span text does not match its substring.")

        # The entity must include its sign, even if '-' is a separate token.
        surface = span["text"]
        starts_with_number = surface[0].isdigit()
        if surface.startswith(".") and len(surface) > 1:
            starts_with_number = surface[1].isdigit()
        if label in NUMERIC_LABELS and starts_with_number:
            prefix = text[:start].rstrip()
            if prefix.endswith(("+", "-", "−", "–", "—")):
                raise ValueError(f"{message}: span excludes a preceding numeric sign.")

        value = span["normalized_value"]
        if value is not None:
            if label in NUMERIC_LABELS:
                if type(value) not in (int, float):
                    raise ValueError(f"{message}: span needs a finite numeric value.")
                try:
                    finite = isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError(f"{message}: span needs a finite numeric value.")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{message}: span needs a categorical string.")

        column = SLOT_BY_LABEL[label]
        if column in record:
            expected = record[column]
            if expected == "":
                expected = None
            if not phase1_audit.values_match(expected, value):
                raise ValueError(f"{message}: span normalization differs from its source.")
        saved_spans.append([label, start, end])
        previous_end = end

    if saved_spans != bio_spans:
        raise ValueError(f"{message}: spans do not match the BIO entity boundaries.")
    # Null values and implicit slots are allowed. Do not invent or remove them.


def validate_navigation_record(record):
    """Retain negative examples while checking executable destination labels."""
    if record.get("intent") == "NAVIGATE_TO_LOCATION" and record.get("capability") != "NAVIGATION_COMMAND":
        raise ValueError(f"Example {record['example_id']}: navigation intent needs NAVIGATION_COMMAND capability.")
    if record.get("capability") != "NAVIGATION_COMMAND":
        return
    message = f"Example {record['example_id']}"
    try:
        frame = json.loads(record["semantic_frame_json"])
        json.dumps(frame, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{message}: invalid navigation semantic frame.") from error
    if not isinstance(frame, dict) or not isinstance(frame.get("parameters"), dict):
        raise ValueError(f"{message}: navigation frame needs a parameters dictionary.")
    if frame.get("intent") != record["intent"]:
        raise ValueError(f"{message}: navigation frame intent differs from the row.")

    expected = record["location"] or None
    if frame["parameters"].get("location") != expected:
        raise ValueError(f"{message}: frame location differs from the source column.")
    if record["intent"] == "UNKNOWN":
        if record["expected_decision"] != "REJECT" or frame["parameters"]:
            raise ValueError(f"{message}: unsupported navigation must remain rejected.")
    elif record["intent"] != "NAVIGATE_TO_LOCATION":
        raise ValueError(f"{message}: unsupported navigation intent.")
    elif frame.get("capability") != "NAVIGATION_COMMAND":
        raise ValueError(f"{message}: navigation frame capability is inconsistent.")

    if record["expected_decision"] == "VALID":
        if expected not in SUPPORTED_LOCATIONS:
            raise ValueError(f"{message}: valid navigation needs HOME or LABORATORY.")
        spans = [s for s in record["slot_spans"] if s["label"] == "LOCATION"]
        if len(spans) != 1 or spans[0]["normalized_value"] != expected:
            raise ValueError(f"{message}: valid navigation needs one matching LOCATION span.")


def validate_workbook_metadata(data, expected_splits=None):
    """Check a supported corpus and keep whole wording groups in one split."""
    required = list(SOURCE_COLUMNS) + list(JSON_COLUMNS.values()) + [
        "annotation_status", "annotation_notes", "location", "navigation_case", "template_family",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing workbook columns: {missing}")
    if expected_splits is None:
        expected_splits = next((counts for counts in CORPUS_LAYOUTS.values()
                                if sum(counts.values()) == len(data)), None)
    if expected_splits is None or len(data) != sum(expected_splits.values()):
        sizes = ", ".join(f"{sum(counts.values()):,}" for counts in CORPUS_LAYOUTS.values())
        raise ValueError(f"Expected a supported corpus size ({sizes}); found {len(data)} rows.")
    for column in phase1_audit.NON_NULL_COLUMNS:
        if not data[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Missing or invalid text in {column}.")
    if data["example_id"].duplicated().any():
        raise ValueError("Duplicate example IDs found.")
    normalized = data["utterance"].str.casefold().str.replace(r"[\W_]+", "", regex=True)
    if normalized.duplicated().any():
        raise ValueError("Duplicate utterances found after case/punctuation normalization.")
    if not data["intent"].isin(INTENT_LABELS).all():
        raise ValueError("Unsupported intent label.")
    if not data["expected_decision"].isin(DECISION_LABELS).all():
        raise ValueError("Unsupported expected decision.")
    if not data["annotation_status"].isin(["AUTO_VERIFIED", "NEEDS_REVIEW"]).all():
        raise ValueError("Unsupported annotation status.")
    if set(data["split"]) != set(expected_splits):
        raise ValueError("Only train, validation and test splits are allowed.")
    for split, expected in expected_splits.items():
        actual = int(data["split"].eq(split).sum())
        if actual != expected:
            raise ValueError(f"Split {split!r}: expected {expected}, found {actual}.")
    for column in ("paraphrase_group_id", "meaning_id", "template_family"):
        present = data[data[column] != ""]
        if present.groupby(column)["split"].nunique().gt(1).any():
            raise ValueError(f"Some {column} values cross split boundaries.")
    if not data.groupby("paraphrase_group_id").size().eq(5).all():
        raise ValueError("Every paraphrase group must contain five rows.")


def read_slot_workbook(workbook_path=DATASET_PATH):
    """Read the full corpus sheet without modifying the Excel file."""
    path = Path(workbook_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Annotated workbook not found: {path}")
    with pd.ExcelFile(path) as workbook:
        sheets = [name for name in CORPUS_LAYOUTS if name in workbook.sheet_names]
        if len(sheets) != 1:
            raise ValueError("Workbook must contain exactly one supported corpus sheet.")
        sheet = sheets[0]
        data = pd.read_excel(workbook, sheet_name=sheet, keep_default_na=False)
    validate_workbook_metadata(data, CORPUS_LAYOUTS[sheet])
    return data


def load_slot_records(workbook_path=DATASET_PATH, split="train"):
    """Load train or validation in original order, retaining review rows."""
    if split not in ("train", "validation"):
        raise ValueError("Development loading permits only train or validation.")
    data = read_slot_workbook(workbook_path)
    selected = data[data["split"] == split]
    records = []
    for _, row in selected.iterrows():
        record = row_to_record(row)
        validate_record(record)
        records.append(record)
    return records


def audit_slot_annotations(workbook_path=DATASET_PATH, reference_path=None):
    """Check all annotations mechanically and return counts, not test examples."""
    data = read_slot_workbook(workbook_path)
    reference_match = None
    reference_rows = None
    reference_columns = None
    policy_updates = 0
    if reference_path is not None:
        with pd.ExcelFile(Path(reference_path).expanduser()) as workbook:
            sheets = [name for name in ("Corpus_4500", *CORPUS_LAYOUTS) if name in workbook.sheet_names]
            if len(sheets) != 1:
                raise ValueError("Reference needs one supported corpus sheet.")
            sheet = sheets[0]
            original = pd.read_excel(workbook, sheet_name=sheet, keep_default_na=False)
        columns = list(original.columns)
        if not set(SOURCE_COLUMNS).issubset(columns) or not set(columns).issubset(data.columns):
            raise ValueError("Reference columns must be present in the Phase 2 workbook.")
        if len(original) > len(data):
            raise ValueError("Reference has more rows than the Phase 2 workbook.")
        current = data.iloc[:len(original)][columns].reset_index(drop=True)
        changed = current.ne(original).any(axis=1)
        reference_match = not bool(changed.any())
        # The expanded edition explicitly changes the generic sensor-check policy.
        # Report those changes instead of claiming byte-for-byte label preservation.
        permitted = {"intent", "diagnostic_property", "expected_decision", "missing_slots",
                     "semantic_frame_json", "validation_reason", "expected_system_behavior", "annotation_notes"}
        for index in original.index[changed]:
            old, new = original.loc[index], current.loc[index]
            fields = set(current.columns[current.loc[index].ne(original.loc[index])])
            if not (fields <= permitted and old["intent"] == "UNKNOWN"
                    and old["capability"] == "SENSOR_DIAGNOSIS"
                    and old["expected_decision"] == "NEEDS_CLARIFICATION"
                    and bool(old["sensor"]) and new["intent"] == "CHECK_SENSOR_HEALTH"
                    and new["diagnostic_property"] == "HEALTH" and new["expected_decision"] == "VALID"):
                raise ValueError("Source content differs beyond the documented sensor-health policy update.")
            old_frame = json.loads(old["semantic_frame_json"])
            new_frame = json.loads(new["semantic_frame_json"])
            expected_frame = dict(old_frame, intent="CHECK_SENSOR_HEALTH",
                                  parameters={"sensor": old["sensor"], "diagnostic_property": "HEALTH"})
            if new_frame != expected_frame:
                raise ValueError("Sensor-health policy update contains an unexpected semantic change.")
            policy_updates += 1
        reference_rows, reference_columns = len(original), len(columns)

    for row_index, row in data.iterrows():
        try:
            validate_record(row_to_record(row))
        except ValueError as error:
            if row["split"] == "test":
                raise ValueError(
                    f"Test structural audit failed at Excel row {row_index + 2}; "
                    "example details are withheld."
                ) from None
            raise ValueError(f"Excel row {row_index + 2}: {error}") from error

    status_counts = {}
    for split in ("train", "validation", "test"):
        selected = data[data["split"] == split]
        status_counts[split] = selected["annotation_status"].value_counts().to_dict()
    return {
        "rows_checked": len(data),
        "split_counts": data["split"].value_counts().to_dict(),
        "annotation_status_by_split": status_counts,
        "original_28_columns_match_reference": reference_match,
        "reference_rows_checked": reference_rows,
        "reference_columns_checked": reference_columns,
        "reference_sensor_health_policy_updates": policy_updates,
        "intent_count": int(data["intent"].nunique()),
        "entity_type_count": len(SLOT_BY_LABEL),
        "bio_label_count": len(ALLOWED_BIO_TAGS),
        "supported_locations": list(SUPPORTED_LOCATIONS),
        "navigation_cases": data.loc[data["capability"] == "NAVIGATION_COMMAND", "navigation_case"].value_counts().to_dict(),
        "structural_audit": "PASS",
        "note": "Automatic integrity checks only; human annotation review is separate.",
    }

def build_label_mappings():
    """Give each BIO label a fixed number."""
    ordered_labels = ["O"] + sorted(label for label in ALLOWED_BIO_TAGS if label != "O")
    label2id = {}
    id2label = {}
    for number, label in enumerate(ordered_labels):
        label2id[label] = number
        id2label[number] = label
    return label2id, id2label


def load_slot_tokenizer(model_name=SLOT_MODEL_NAME, local_files_only=False):
    """Load a fast tokenizer only when model preprocessing is requested."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError("Tokenization needs transformers in your active Python environment.") from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"Tokenizer {model_name} is not a fast tokenizer.")

    return tokenizer


def align_word_labels(word_ids, bio_tags, label2id):
    """Label the first subword; ignore special tokens and repeated subwords."""
    for tag in bio_tags:
        if tag not in label2id:
            raise ValueError(f"BIO label is missing from label2id: {tag}")
    labels = []
    previous_word_id = -1
    for word_id in word_ids:
        if word_id is None:
            labels.append(IGNORE_LABEL_ID)
        elif type(word_id) is not int or not 0 <= word_id < len(bio_tags):
            raise ValueError("Tokenizer word ID is outside the annotation range.")
        elif word_id == previous_word_id:
            labels.append(IGNORE_LABEL_ID)
        else:
            if word_id != previous_word_id + 1:
                raise ValueError("Tokenizer skipped or reordered an annotated word.")
            labels.append(label2id[bio_tags[word_id]])
            previous_word_id = word_id
    if previous_word_id != len(bio_tags) - 1:
        raise ValueError("Tokenizer did not cover all annotated words; check truncation.")
    return labels


def tokenize_slot_records(records, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """Return unpadded model features and refuse to silently truncate labels."""
    if not tokenizer.is_fast:
        raise ValueError("Word alignment requires a fast tokenizer.")
    if type(max_length) is not int or max_length < 3:
        raise ValueError("max_length must be an integer of at least 3.")
    if max_length > tokenizer.model_max_length:
        raise ValueError("max_length exceeds this tokenizer's model limit.")
    label2id, _ = build_label_mappings()
    features = []
    for record in records:
        validate_record(record)
        encoded = tokenizer(record["tokens"], is_split_into_words=True,
                            truncation=False, padding=False)
        if len(encoded["input_ids"]) > max_length:
            raise ValueError(
                f"Example {record['example_id']}: {len(encoded['input_ids'])} tokens "
                f"exceed max_length={max_length}; increase it before training.")
        labels = align_word_labels(encoded.word_ids(), record["bio_tags"], label2id)
        feature = {name: encoded[name] for name in tokenizer.model_input_names if name in encoded}
        feature["labels"] = labels
        if len(feature["input_ids"]) != len(labels):
            raise ValueError(f"Example {record['example_id']}: input and label lengths differ.")
        features.append(feature)
    return features


def main():
    """Run a data check from the terminal without writing any source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook_path", nargs="?", default=str(DATASET_PATH))
    parser.add_argument("--split", choices=["train", "validation"])
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--reference", help="Original workbook, used with --audit.")
    parser.add_argument("--tokenize", action="store_true", help="Prepare model inputs; no training.")
    parser.add_argument("--show-example", action="store_true", help="Print one development example with aligned subwords.")
    parser.add_argument("--model-name", default=SLOT_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=MAX_SEQUENCE_LENGTH)
    args = parser.parse_args()
    if args.audit and args.split:
        parser.error("Choose --audit or --split, not both.")
    if args.reference and not args.audit:
        parser.error("--reference requires --audit.")
    if args.audit and (args.tokenize or args.show_example):
        parser.error("Choose --audit or tokenization, not both.")
    if args.show_example and not args.tokenize:
        parser.error("--show-example requires --tokenize.")
    if args.audit:
        print(json.dumps(audit_slot_annotations(args.workbook_path, args.reference), indent=2))
    else:
        split = args.split or "train"
        records = load_slot_records(args.workbook_path, split)
        print("Split:", split)
        print("Records loaded:", len(records))
        print("BIO labels:", len(ALLOWED_BIO_TAGS))
        print("Supported locations:", ", ".join(SUPPORTED_LOCATIONS))
        if args.tokenize:
            tokenizer = load_slot_tokenizer(args.model_name, args.local_files_only)
            features = tokenize_slot_records(records, tokenizer, args.max_length)
            print("Tokenized records:", len(features))
            print("Longest input:", max(len(item["input_ids"]) for item in features))
            print("Alignment: PASS")
            if args.show_example:
                index = next((i for i, r in enumerate(records) if r["intent"] == "NAVIGATE_TO_LOCATION"), 0)
                feature = features[index]
                _, id2label = build_label_mappings()
                print("Example:", records[index]["example_id"], records[index]["utterance"])
                pieces = tokenizer.convert_ids_to_tokens(feature["input_ids"])
                for piece, label_id in zip(pieces, feature["labels"]):
                    tag = "IGNORE (-100)" if label_id == IGNORE_LABEL_ID else id2label[label_id]
                    print(f"  {piece:20s} {tag}")


if __name__ == "__main__":
    main()
