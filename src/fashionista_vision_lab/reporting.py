"""Render saved runs without importing or contacting model runtimes."""

import json
from pathlib import Path


def shown(value):
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def render_report(paths: list[Path]) -> str:
    runs = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        runs.append((manifest, metrics))
    if not runs:
        raise ValueError("at least one completed run is required")
    keys = {manifest["comparison_key"] for manifest, _ in runs}
    if len(runs) > 1 and len(keys) != 1:
        raise ValueError("Runs have different dataset/image hashes, tasks or scoring policies; compare only equivalent runs.")
    lines = ["# Fashionista Vision Lab report", "",
             "Scores describe these inputs and this protocol only. N/A means no eligible denominator.",
             "Failures remain in eligible accuracy/recall denominators. Check failure counts and exclusions.", ""]
    for manifest, metrics in runs:
        lines.extend([f"## {manifest['model']} — {manifest['task']}", "",
                      f"Run: `{manifest['run_id']}`", "",
                      f"Data status: **{manifest['data_status']}**", "",
                      f"Attempts: {metrics['attempted']}; failures: {metrics['failed']}; excluded: {len(manifest.get('excluded', []))}.", "",
                      f"Mean wall latency (includes first lazy load): {shown(metrics['mean_latency_seconds'])} seconds.", "",
                      f"Tokens reported: input {metrics['input_tokens']}, output {metrics['output_tokens']}, thinking {metrics['thinking_tokens']}; usage coverage {metrics['usage_records']}/{metrics['attempted']}.", "",
                      f"Estimated cost: {shown(metrics['estimated_cost'])} {metrics.get('cost_currency') or ''}; priced records {metrics['priced_records']}/{metrics['attempted']}.", ""])
        if manifest["task"] == "detection":
            lines.extend([f"Mode: {manifest['protocol']['detection_mode']}", "",
                          "| Precision | Recall | Matched IoU | TP | FP | FN |", "| --- | --- | --- | --- | --- | --- |",
                          "| " + " | ".join(shown(metrics[k]) for k in ("precision", "recall", "mean_matched_iou", "tp", "fp", "fn")) + " |", ""])
        elif manifest["task"] == "attributes":
            lines.extend(["Ground-truth crops (or unambiguous single-garment images), not end-to-end detection.", "",
                          "| Field | Accuracy | Eligible | Missing | Incorrect | Unsupported values | Unscored predictions |",
                          "| --- | --- | --- | --- | --- | --- | --- |"])
            for field, counts in metrics["per_attribute"].items():
                lines.append("| " + field + " | " + " | ".join(shown(counts[k]) for k in (
                    "accuracy", "eligible", "missing", "incorrect", "unsupported_values", "unscored_prediction")) + " |")
            lines.append("")
        if metrics.get("failure_codes"):
            lines.extend(["Failure codes: `" + json.dumps(metrics["failure_codes"], sort_keys=True) + "`", ""])
        lines.extend(["Exclusions and exact configuration are in manifest.json; per-sample errors and scores are in normalized.jsonl.", ""])
    return "\n".join(lines)
