from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import sys


TERMS = {
    "has_exception_term": re.compile(r"\bexception\b", re.I),
    "has_failed_term": re.compile(r"\bfailed\b", re.I),
    "has_timeout_term": re.compile(r"\btimeout\b", re.I),
    "has_error_term": re.compile(r"\berror\b", re.I),
}

def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def main() -> None:
    configure_csv_field_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_traces_anonymized.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/microservices_sample/microsservices_lab_trace_features.csv"))
    args = parser.parse_args()
    traces: dict[str, dict] = defaultdict(lambda: {
        "event_count": 0, "info_count": 0, "warn_count": 0, "error_count": 0,
        "other_count": 0,
        "has_stack_trace": 0, "services": set(), **{name: 0 for name in TERMS},
    })
    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            trace_id = (row.get("tc_trace_id") or "").strip()
            if not trace_id:
                continue
            item = traces[trace_id]
            item["event_count"] += 1
            level = (row.get("level") or "").strip().upper()
            level_column = {"INFO": "info_count", "WARN": "warn_count", "ERROR": "error_count"}.get(level, "other_count")
            item[level_column] += 1
            if (row.get("stack_trace") or "").strip():
                item["has_stack_trace"] = 1
            service = (row.get("tc_service") or "").strip()
            if service:
                item["services"].add(service)
            text = f"{row.get('message') or ''} {row.get('stack_trace') or ''}"
            for name, pattern in TERMS.items():
                item[name] = int(bool(item[name] or pattern.search(text)))
    fields = ["tc_trace_id", "event_count", "info_count", "warn_count", "error_count", "other_count",
              "error_rate", "warn_rate", "has_stack_trace", "service_count", *TERMS,
              "text_signal_count", "has_any_signal"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trace_id in sorted(traces):
            item = traces[trace_id]
            text_count = sum(item[name] for name in TERMS)
            writer.writerow({"tc_trace_id": trace_id, "event_count": item["event_count"],
                "info_count": item["info_count"], "warn_count": item["warn_count"],
                "error_count": item["error_count"], "error_rate": item["error_count"] / item["event_count"],
                "other_count": item["other_count"],
                "warn_rate": item["warn_count"] / item["event_count"],
                "has_stack_trace": item["has_stack_trace"], "service_count": len(item["services"]),
                **{name: item[name] for name in TERMS}, "text_signal_count": text_count,
                "has_any_signal": int(item["error_count"] > 0 or item["warn_count"] > 0 or
                                       item["has_stack_trace"] or text_count > 0)})
    manifest = {"stage": 2, "stage_name": "build_trace_features", "input": str(args.input),
                "output": str(args.output), "traces": len(traces),
                "labels_in_features": False, "feature_columns": fields[1:]}
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
