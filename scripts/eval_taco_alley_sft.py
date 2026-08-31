#!/usr/bin/env python3
"""Evaluate a Taco Alley SFT GGUF model with llama.cpp."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SEED = 42
EVAL_SAMPLE_SIZE = 120
INPUT_FIELDS = ["customer_id", "date", "location", "message"]
OUTPUT_FIELDS = ["category", "sub_category", "tone_urgency"]
REQUIRED_FIELDS = [*INPUT_FIELDS, *OUTPUT_FIELDS]
TONE_ALLOWED = {"low", "moderate", "high", "urgent"}
SYSTEM_PROMPT = (
    "You are a complaint structuring assistant for Taco Alley. "
    "Input is a JSON object with fields customer_id, date, location, message. "
    "Return only valid JSON with exactly these fields in the output: "
    f"{REQUIRED_FIELDS}. "
    "Keep the original input fields unchanged and add correct values for category, "
    "sub_category, and tone_urgency. Do not add any extra keys or commentary."
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Taco Alley SFT GGUF model with llama.cpp.")
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        help="Path to the GGUF model (required unless --server-url is used).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "taco_alley_customer_complaints.csv",
        help="Complaint CSV path (default: repository dataset).",
    )
    parser.add_argument("--llama-cli", default="llama-cli", help="llama-cli command or path.")
    parser.add_argument(
        "--server-url",
        help="llama-server base URL, such as http://127.0.0.1:8080. Keeps the model loaded between samples.",
    )
    parser.add_argument("--context-size", "-c", type=int, default=1024)
    parser.add_argument("--max-new-tokens", "-n", type=int, default=220)
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=999,
        help="Number of layers to offload in direct llama-cli mode (default: 999).",
    )
    parser.add_argument("--request-timeout", type=int, default=300, help="Server request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=EVAL_SAMPLE_SIZE, help="Maximum samples to evaluate.")
    parser.add_argument("--output", type=Path, default=Path("taco_alley_sft_eval.json"))
    return parser.parse_args()


def load_examples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        raw_rows = list(csv.DictReader(source))
    if not raw_rows:
        raise ValueError("Dataset contains no rows.")
    missing_columns = [field for field in REQUIRED_FIELDS if field not in raw_rows[0]]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    examples = []
    for row in raw_rows:
        record = {field: str(row.get(field, "")).strip() for field in REQUIRED_FIELDS}
        if any(not record[field] for field in REQUIRED_FIELDS):
            continue
        if not re.match(r"^CUST-\d{5,}", record["customer_id"]):
            continue
        if record["tone_urgency"].lower() not in TONE_ALLOWED:
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", record["date"]):
            continue
        if record["date"] > time.strftime("%Y-%m-%d"):
            continue
        examples.append(
            {
                "input_obj": {field: record[field] for field in INPUT_FIELDS},
                "target_obj": record,
            }
        )
    if not examples:
        raise ValueError("No valid examples remain after notebook-equivalent validation.")
    return examples


def parse_json_from_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in model output.") from None
        return json.loads(cleaned[start : end + 1])


def generate_structured_prediction(arguments: argparse.Namespace, input_obj: dict[str, str]) -> tuple[dict[str, str] | None, str]:
    prompt = json.dumps(input_obj, ensure_ascii=False)
    if arguments.server_url:
        return generate_server_prediction(arguments, prompt)
    command = [
        arguments.llama_cli,
        "-m",
        str(arguments.model),
        "-p",
        prompt,
        "-sys",
        SYSTEM_PROMPT,
        "-cnv",
        "-n",
        str(arguments.max_new_tokens),
        "-c",
        str(arguments.context_size),
        "-s",
        str(SEED),
        "-ngl",
        str(arguments.gpu_layers),
        "--temp",
        "0",
        "--no-display-prompt",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    raw_text = completed.stdout.strip()
    if completed.returncode != 0:
        return None, f"llama-cli failed ({completed.returncode}): {completed.stderr.strip()}"
    try:
        parsed = parse_json_from_output(raw_text)
        return {field: str(parsed.get(field, "")).strip() for field in REQUIRED_FIELDS}, raw_text
    except (ValueError, json.JSONDecodeError):
        return None, raw_text


def generate_server_prediction(arguments: argparse.Namespace, prompt: str) -> tuple[dict[str, str] | None, str]:
    endpoint = f"{arguments.server_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": arguments.max_new_tokens,
            "seed": SEED,
        }
    ).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=arguments.request_timeout) as response:
            response_json = json.loads(response.read().decode("utf-8"))
        raw_text = response_json["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as error:
        return None, f"llama-server request failed: {error}"
    try:
        parsed = parse_json_from_output(raw_text)
        return {field: str(parsed.get(field, "")).strip() for field in REQUIRED_FIELDS}, raw_text
    except (ValueError, json.JSONDecodeError):
        return None, raw_text


def evaluate_model(arguments: argparse.Namespace, examples: list[dict[str, Any]]) -> dict[str, Any]:
    subset = examples[: arguments.limit]
    rows = []
    started = time.perf_counter()
    print(f"Evaluating {arguments.model} on {len(subset)} samples...")
    for index, example in enumerate(subset, start=1):
        prediction, raw_text = generate_structured_prediction(arguments, example["input_obj"])
        gold = example["target_obj"]
        schema_valid = 1.0
        if prediction is None:
            schema_valid = category_match = sub_category_match = tone_match = 0.0
        else:
            if any(not prediction[field] for field in REQUIRED_FIELDS):
                schema_valid = 0.0
            if not prediction["customer_id"].startswith("CUST-"):
                schema_valid = 0.0
            if prediction["tone_urgency"].lower() not in TONE_ALLOWED:
                schema_valid = 0.0
            category_match = float(prediction["category"] == gold["category"])
            sub_category_match = float(prediction["sub_category"] == gold["sub_category"])
            tone_match = float(prediction["tone_urgency"] == gold["tone_urgency"])
        rows.append(
            {
                "schema_valid": schema_valid,
                "category_match": category_match,
                "sub_category_match": sub_category_match,
                "tone_match": tone_match,
                "weighted_score": 0.6 * schema_valid + 0.3 * category_match + 0.1 * tone_match,
                "raw_preview": raw_text[:220].replace("\n", " "),
            }
        )
        if index % 25 == 0 or index == len(subset):
            print(f"  Processed {index}/{len(subset)} samples")
    elapsed = time.perf_counter() - started
    return {
        "model": str(arguments.model),
        "samples": len(subset),
        "seconds": elapsed,
        "samples_per_second": len(subset) / elapsed if elapsed else 0.0,
        "schema_valid_rate": statistics.mean(row["schema_valid"] for row in rows),
        "category_accuracy": statistics.mean(row["category_match"] for row in rows),
        "sub_category_accuracy": statistics.mean(row["sub_category_match"] for row in rows),
        "tone_accuracy": statistics.mean(row["tone_match"] for row in rows),
        "weighted_score": statistics.mean(row["weighted_score"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    arguments = parse_arguments()
    if not arguments.dataset.is_file() or arguments.limit < 1:
        print("Error: dataset must exist, and --limit must be positive.", file=sys.stderr)
        return 2
    if not arguments.server_url and (arguments.model is None or not arguments.model.is_file()):
        print("Error: a valid model path is required unless --server-url is used.", file=sys.stderr)
        return 2
    try:
        metrics = evaluate_model(arguments, load_examples(arguments.dataset))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    arguments.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nEvaluation summary:")
    for key, value in metrics.items():
        if key != "rows":
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Detailed results written to: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())