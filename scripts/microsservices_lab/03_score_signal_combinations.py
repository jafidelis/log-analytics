from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(a == b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(y_true), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "coverage": sum(y_pred) / len(y_pred) if y_pred else 0.0}


def main() -> None:
    configure_csv_field_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path(
        "data/processed/microservices_sample/microsservices_lab_trace_features.csv"))
    parser.add_argument("--labels", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_trace_manual_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage3_signal_scores.json"))
    args = parser.parse_args()
    labels = {}
    with args.labels.open("r", encoding="utf-8-sig", newline="") as stream:
        labels = {r["tc_trace_id"].strip(): r["label"].strip() for r in csv.DictReader(stream)}
    rows = []
    with args.features.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["tc_trace_id"] in labels and labels[r["tc_trace_id"]] != "incerto"]
    rules = {
        "error": lambda r: int(float(r["error_count"]) > 0),
        "warn": lambda r: int(float(r["warn_count"]) > 0),
        "stack_trace": lambda r: int(float(r["has_stack_trace"]) > 0),
        "text": lambda r: int(float(r["text_signal_count"]) > 0),
        "error_or_warn": lambda r: int(float(r["error_count"]) > 0 or float(r["warn_count"]) > 0),
        "error_or_stack": lambda r: int(float(r["error_count"]) > 0 or float(r["has_stack_trace"]) > 0),
        "any_signal": lambda r: int(float(r["has_any_signal"]) > 0),
    }
    y = [int(labels[r["tc_trace_id"]] == "anomalia") for r in rows]
    scored = {name: metrics(y, [fn(r) for r in rows]) for name, fn in rules.items()}
    manifest = {"stage": 3, "stage_name": "score_signal_combinations", "labeled_rows_used": len(rows),
                "incerto_excluded": True, "labels_used_for_model_fit": False, "scores": scored}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
