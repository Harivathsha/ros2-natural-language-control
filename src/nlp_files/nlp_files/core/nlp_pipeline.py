"""Learned text-to-command inference. No keyword, alias or spelling rules."""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nlp_files.config import (
    MAX_ANGLE_DEG, MAX_DISTANCE_M, MAX_SPEED_MPS, MIN_CONFIDENCE,
    MODEL_PATH, ROUTE_BY_CAPABILITY,
)


INPUT_PREFIX = "parse robot command: "
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 256


def read_prediction(text):
    """Parse strict JSON; never repair or reinterpret generated values."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate generated JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or set(value) != {"intent", "parameters", "decision"}:
        raise ValueError("The model did not produce the expected command object.")
    return value


def validate_prediction(prediction, schema):
    """Check output types, supported robot interfaces and physical limits.

    This function receives no command text. It cannot match input keywords,
    correct spellings, choose a destination or convert a unit.
    """
    empty = {"schema_version": "1.0", "capability": "ANY", "intent": "UNKNOWN", "parameters": {}}

    def decision(label, reason, missing=()):
        return {"decision": label, "reason": reason, "missing_slots": list(missing)}

    if not isinstance(prediction, dict):
        return empty, decision("REJECT", "The model output is not a command object.")
    intent = prediction.get("intent")
    parameters = prediction.get("parameters")
    learned_decision = prediction.get("decision")
    if not isinstance(intent, str) or intent not in schema["intents"]:
        return empty, decision("REJECT", "The model predicted an unsupported intent.")
    specification = schema["intents"][intent]
    frame = {"schema_version": "1.0", "capability": specification["capability"],
             "intent": intent, "parameters": parameters if isinstance(parameters, dict) else {}}
    if learned_decision not in ("VALID", "NEEDS_CLARIFICATION", "REJECT"):
        return frame, decision("REJECT", "The model predicted an invalid decision.")
    if not isinstance(parameters, dict) or set(parameters) - set(specification["parameters"]):
        return frame, decision("REJECT", "The generated parameters do not match the robot interface.")
    for name, value in parameters.items():
        expected = specification["parameters"][name]
        if expected["type"] == "number":
            if type(value) not in (int, float) or not math.isfinite(value):
                return frame, decision("REJECT", f"{name} must be a finite number.")
        elif not isinstance(value, str):
            return frame, decision("REJECT", f"{name} must be a canonical string value.")
        elif "values" in expected and value not in expected["values"]:
            return frame, decision("REJECT", f"The generated {name} is unsupported.")

    missing = [key for key in specification["required"] if key not in parameters]
    if learned_decision == "REJECT":
        return frame, decision("REJECT", "The model classified this request as unsupported or invalid.")
    if learned_decision == "NEEDS_CLARIFICATION":
        return frame, decision("NEEDS_CLARIFICATION", "The model requests clarification.", missing)
    if intent == "UNKNOWN":
        return frame, decision("REJECT", "An UNKNOWN request cannot be executed.")
    if missing:
        return frame, decision("NEEDS_CLARIFICATION", "The generated command is incomplete.", missing)

    for name, maximum in (("distance_m", MAX_DISTANCE_M), ("speed_mps", MAX_SPEED_MPS),
                          ("angle_deg", MAX_ANGLE_DEG)):
        if name in parameters and not 0 < parameters[name] <= maximum:
            return frame, decision("REJECT", f"{name} must be greater than 0 and at most {maximum}.")
    if "threshold_value" in parameters and parameters["threshold_value"] <= 0:
        return frame, decision("REJECT", "threshold_value must be positive.")
    # These are robot API units, not input-word conversion rules.
    units = {"MINIMUM_OBSTACLE_DISTANCE_M": "m", "PUBLISH_RATE_HZ": "Hz", "MESSAGE_AGE_S": "s"}
    if "condition_property" in parameters and parameters.get("threshold_unit") != units.get(parameters["condition_property"]):
        return frame, decision("REJECT", "The generated threshold unit does not match its property.")
    if "location" in parameters and parameters["location"] not in schema["supported_locations"]:
        return frame, decision("REJECT", "The destination is outside the configured robot locations.")
    if "rule_id" in parameters:
        identifier = parameters["rule_id"]
        if not identifier.startswith("R") or not identifier[1:].isdigit():
            return frame, decision("REJECT", "The generated rule ID is malformed.")
    return frame, decision("VALID", "The predicted command matches the robot interface and limits.")


@torch.inference_mode()
def generate_predictions(model, tokenizer, texts, device):
    """Generate normalized command JSON directly from untouched input text."""
    encoded = tokenizer([INPUT_PREFIX + text for text in texts], padding=True, truncation=False,
                        return_token_type_ids=False, return_tensors="pt")
    if encoded["input_ids"].shape[1] > MAX_INPUT_LENGTH:
        raise ValueError("Please shorten the command; the input exceeds the model's training length.")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    outputs = model.generate(
        **encoded, max_new_tokens=MAX_OUTPUT_LENGTH, do_sample=False, num_beams=1,
        return_dict_in_generate=True, output_scores=True,
    )
    strings = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    token_ids = outputs.sequences[:, -scores.shape[1]:]
    confidences = []
    for row, ids in zip(scores, token_ids):
        keep = ids != tokenizer.pad_token_id
        confidences.append(float(row[keep].mean().exp()) if keep.any() else 0.0)
    return list(zip(strings, confidences))


class NLPPipeline:
    """One fine-tuned encoder-decoder predicts intent, values and decision."""

    def __init__(self, model_path=MODEL_PATH, min_confidence=MIN_CONFIDENCE):
        self.model_path = Path(model_path).expanduser().resolve()
        self.min_confidence = float(min_confidence)
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1.")
        schema_path = self.model_path / "command_schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(
                f"Train train_semantic_model.py first. Expected: {schema_path}. "
                "The earlier intent and BIO checkpoints cannot generate normalized frames."
            )
        self.schema = json.loads(schema_path.read_text())
        if self.schema.get("format_version") != 1:
            raise ValueError("Unsupported semantic-model schema version.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(self.device).eval()

    def process(self, text):
        text = text.strip() if isinstance(text, str) else ""
        raw, confidence, prediction = "", 0.0, None
        error = None
        if not text:
            error = "The command is empty."
        else:
            try:
                raw, confidence = generate_predictions(self.model, self.tokenizer, [text], self.device)[0]
                prediction = read_prediction(raw)
            except ValueError as failure:
                error = str(failure)
        frame, validation = validate_prediction(prediction, self.schema)
        if error:
            validation = {"decision": "REJECT", "reason": error, "missing_slots": []}
        elif confidence < self.min_confidence:
            validation = {"decision": "NEEDS_CLARIFICATION", "reason": "Low model token confidence; please rephrase.", "missing_slots": []}
        route = ROUTE_BY_CAPABILITY.get(frame["capability"], "NO_EXECUTION") if validation["decision"] == "VALID" else "NO_EXECUTION"
        response = (f"Request ready: {frame['intent']} {json.dumps(frame['parameters'])}"
                    if route != "NO_EXECUTION" else validation["reason"])
        return {
            "text": text, "raw_intent": frame["intent"], "selected_intent": frame["intent"],
            "intent_source": "semantic_model", "confidence": round(confidence, 4),
            "confidence_type": "mean_generated_token_probability",
            "model_output": raw, "semantic_frame": frame, "validation": validation,
            "route": route, "response": response,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--text")
    args = parser.parse_args()
    pipeline = NLPPipeline(model_path=args.model_path)
    print("Learned semantic NLP ready. Device:", pipeline.device)
    while True:
        try:
            text = args.text if args.text is not None else input("\nCommand (quit to exit): ")
            if text.strip().lower() in {"quit", "exit"}:
                break
            print(json.dumps(pipeline.process(text), indent=2, ensure_ascii=False))
            if args.text is not None:
                break
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as error:
            print(f"Error: {error}")
            if args.text is not None:
                break


if __name__ == "__main__":
    main()
