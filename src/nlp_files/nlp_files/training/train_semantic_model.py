"""Fine-tune FLAN-T5: command text -> intent, canonical parameters, decision."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from nlp_files.config import MIN_CONFIDENCE, PROJECT_ROOT
from nlp_files.core.nlp_pipeline import (
    INPUT_PREFIX, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH,
    NLPPipeline, generate_predictions, read_prediction, validate_prediction,
)
from nlp_files.data.slot_dataset import DATASET_PATH, SUPPORTED_LOCATIONS, read_slot_workbook


def target_for(row):
    frame = json.loads(row["semantic_frame_json"])
    return {"intent": frame["intent"], "parameters": frame["parameters"],
            "decision": row["expected_decision"]}


def build_schema(train):
    """Derive the output contract from training annotations, never input words."""
    intents = {}
    for intent, rows in train.groupby("intent", sort=True):
        frames = [json.loads(value) for value in rows["semantic_frame_json"]]
        capabilities = {frame["capability"] for frame in frames}
        if len(capabilities) != 1:
            # UNKNOWN clarification examples may retain SENSOR_DIAGNOSIS.
            if intent != "UNKNOWN":
                raise ValueError(f"Inconsistent capabilities for {intent}.")
            capability = "ANY"
        else:
            capability = capabilities.pop()
        valid = [frame["parameters"] for frame, decision in zip(frames, rows["expected_decision"])
                 if decision == "VALID"]
        required = sorted(set.intersection(*(set(item) for item in valid))) if valid else []
        values_by_key = {}
        for frame in frames:
            for name, value in frame["parameters"].items():
                values_by_key.setdefault(name, []).append(value)
        parameters = {}
        for name, values in sorted(values_by_key.items()):
            if all(type(value) in (int, float) and math.isfinite(value) for value in values):
                parameters[name] = {"type": "number"}
            elif all(isinstance(value, str) for value in values):
                parameters[name] = {"type": "string"}
                if name != "rule_id":  # New robot rule identifiers remain possible.
                    parameters[name]["values"] = sorted(set(values))
            else:
                raise ValueError(f"Mixed or invalid annotation types for {intent}.{name}.")
        intents[intent] = {"capability": capability, "required": required, "parameters": parameters}
    return {"format_version": 1, "supported_locations": list(SUPPORTED_LOCATIONS), "intents": intents}


def make_features(rows, tokenizer):
    features = []
    for _, row in rows.iterrows():
        target = json.dumps(target_for(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        feature = tokenizer(INPUT_PREFIX + row["utterance"], truncation=False, return_token_type_ids=False)
        labels = tokenizer(text_target=target, truncation=False)["input_ids"]
        if len(feature["input_ids"]) > MAX_INPUT_LENGTH or len(labels) >= MAX_OUTPUT_LENGTH:
            raise ValueError(f"{row['example_id']}: input or target exceeds model length; no silent truncation.")
        if tokenizer.unk_token_id in labels:
            raise ValueError(f"{row['example_id']}: tokenizer cannot represent a target character.")
        decoded = tokenizer.decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if read_prediction(decoded) != target_for(row):
            raise ValueError(f"{row['example_id']}: tokenizer changes the target JSON values.")
        feature["labels"] = labels
        features.append(feature)
    return features


def prepare_tokenizer(tokenizer, train):
    """Make JSON syntax and otherwise-unrepresentable output values lossless."""
    additions = {"{", "}"}
    for _, row in train.iterrows():
        for value in target_for(row)["parameters"].values():
            if isinstance(value, str):
                quoted = json.dumps(value)
                if tokenizer.unk_token_id in tokenizer(quoted)["input_ids"]:
                    additions.add(quoted)
    tokenizer.add_tokens(sorted(additions))


@torch.inference_mode()
def evaluate(model, tokenizer, validation, schema, device, batch_size, errors_path=None):
    model.eval()
    counts = dict(valid_json=0, intent_correct=0, parameters_exact=0, decision_correct=0,
                  command_exact=0, numeric_exact=0, numeric_rows=0, unsafe_accepts=0,
                  wrong_command_accepts=0)
    errors = []
    records = validation.to_dict("records")
    for start in tqdm(range(0, len(records), batch_size), desc="Validation"):
        rows = records[start:start + batch_size]
        guesses = generate_predictions(model, tokenizer, [row["utterance"] for row in rows], device)
        for row, (raw, confidence) in zip(rows, guesses):
            gold = target_for(row)
            valid_json = True
            try:
                prediction = read_prediction(raw)
                counts["valid_json"] += 1
            except ValueError:
                prediction = {}
                valid_json = False
            _, validation_result = validate_prediction(prediction, schema)
            decision = validation_result["decision"]
            if confidence < MIN_CONFIDENCE:
                decision = "NEEDS_CLARIFICATION"
            same_intent = prediction.get("intent") == gold["intent"]
            same_parameters = prediction.get("parameters") == gold["parameters"]
            same_decision = decision == gold["decision"]
            counts["intent_correct"] += same_intent
            counts["parameters_exact"] += same_parameters
            counts["decision_correct"] += same_decision
            counts["command_exact"] += same_intent and same_parameters and same_decision
            false_accept = decision == "VALID" and gold["decision"] != "VALID"
            wrong_command_accept = decision == "VALID" and not (same_intent and same_parameters)
            counts["unsafe_accepts"] += false_accept
            counts["wrong_command_accepts"] += wrong_command_accept
            if errors_path is not None and not (same_intent and same_parameters and same_decision):
                failures = [name for name, failed in (
                    ("invalid_json", not valid_json), ("intent", not same_intent),
                    ("parameters", not same_parameters), ("decision", not same_decision),
                    ("false_accept", false_accept), ("wrong_command_accept", wrong_command_accept),
                ) if failed]
                errors.append({
                    "example_id": row["example_id"], "utterance": row["utterance"],
                    "expected_intent": gold["intent"], "predicted_intent": prediction.get("intent"),
                    "expected_parameters": json.dumps(gold["parameters"], sort_keys=True),
                    "predicted_parameters": json.dumps(prediction.get("parameters"), sort_keys=True),
                    "expected_decision": gold["decision"], "predicted_decision": decision,
                    "error_types": " | ".join(failures), "false_accept": bool(false_accept),
                    "wrong_command_accept": bool(wrong_command_accept),
                    "token_confidence": round(confidence, 6), "model_output": raw,
                })
            numeric = {key: value for key, value in gold["parameters"].items()
                       if type(value) in (int, float)}
            if numeric:
                predicted = prediction.get("parameters", {})
                counts["numeric_rows"] += 1
                counts["numeric_exact"] += isinstance(predicted, dict) and all(
                    type(predicted.get(key)) in (int, float) and predicted[key] == value
                    for key, value in numeric.items()
                )
    total = len(records)
    metrics = {name: round(counts[name] / total, 4) for name in (
        "valid_json", "intent_correct", "parameters_exact", "decision_correct", "command_exact")}
    metrics["numeric_exact"] = round(counts["numeric_exact"] / max(counts["numeric_rows"], 1), 4)
    metrics["unsafe_accepts"] = counts["unsafe_accepts"]
    metrics["wrong_command_accepts"] = counts["wrong_command_accepts"]
    if errors_path is not None:
        path = Path(errors_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["example_id", "utterance", "expected_intent", "predicted_intent",
                  "expected_parameters", "predicted_parameters", "expected_decision",
                  "predicted_decision", "error_types", "false_accept", "wrong_command_accept",
                  "token_confidence", "model_output"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(errors)
        print(f"Validation errors saved: {path} ({len(errors)} rows)", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", nargs="?", default=str(DATASET_PATH))
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts/semantic_phase2"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate the saved output-dir checkpoint and export validation errors; do not train.")
    parser.add_argument("--errors-file", default=str(PROJECT_ROOT / "outputs/semantic_validation_errors.csv"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive.")
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    data = read_slot_workbook(args.workbook)
    train, validation = data[data["split"] == "train"], data[data["split"] == "validation"]
    if args.evaluate_only:
        pipeline = NLPPipeline(model_path=args.output_dir)
        metrics = evaluate(pipeline.model, pipeline.tokenizer, validation, pipeline.schema,
                           pipeline.device, args.batch_size, errors_path=args.errors_file)
        print("RELOADED CHECKPOINT:", json.dumps(metrics), flush=True)
        return
    schema = build_schema(train)
    print(f"Training records: {len(train)} | Validation records: {len(validation)}", flush=True)
    print("Targets: learned intent + canonical values + decision; no language dictionaries.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    prepare_tokenizer(tokenizer, train)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    train_features = make_features(train, tokenizer)
    make_features(validation, tokenizer)  # Check every validation target is representable.
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.output_dir).expanduser().resolve()
    best = (-1.0, -float("inf"))
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_tokens = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; training stopped without saving this epoch.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int((batch["labels"] != -100).sum())
            total_loss += loss.item() * tokens
            total_tokens += tokens
        metrics = evaluate(model, tokenizer, validation, schema, device, args.batch_size)
        print(f"Epoch {epoch}: train_loss={total_loss / total_tokens:.4f} | {json.dumps(metrics)}", flush=True)
        score = (metrics["command_exact"], -metrics["unsafe_accepts"])
        if score > best:
            best = score
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "command_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
            print("Saved best semantic model:", output_dir, flush=True)
    print("Best validation command exact match:", best[0])
    print("Model:", output_dir)
    print("Token confidence is not calibrated command correctness. Evaluate new commands before ROS execution.")


if __name__ == "__main__":
    main()
