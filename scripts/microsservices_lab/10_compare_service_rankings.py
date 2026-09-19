from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import re


SIGNAL_TERMS = re.compile(r"\b(exception|failed|timeout|error)\b", re.I)


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts", type=Path, default=Path(
        "artifacts/microsservices_lab/stage8_trace_alerts.csv"))
    parser.add_argument("--events", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_traces_anonymized.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage10_service_ranking_comparison.json"))
    parser.add_argument("--candidates-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage10_service_ranking_candidates.csv"))
    args = parser.parse_args()

    alerts = {}
    with args.alerts.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("alert_level") != "none":
                alerts[row["tc_trace_id"].strip()] = row
    global_service_events = Counter()
    groups: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "event_count": 0, "error_count": 0, "stack_count": 0,
        "text_count": 0, "first_timestamp": "", "first_event_index": 0,
        "first_signal_index": 0, "callstack_hits": 0,
    }))
    event_index = 0
    with args.events.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            event_index += 1
            service = (row.get("tc_service") or "").strip() or "<UNKNOWN_SERVICE>"
            global_service_events[service] += 1
            trace_id = (row.get("tc_trace_id") or "").strip()
            if trace_id not in alerts:
                continue
            item = groups[trace_id][service]
            item["event_count"] += 1
            level = (row.get("level") or "").strip().upper()
            has_stack = bool((row.get("stack_trace") or "").strip())
            has_text = bool(SIGNAL_TERMS.search(f"{row.get('message') or ''} {row.get('stack_trace') or ''}"))
            item["error_count"] += int(level == "ERROR")
            item["stack_count"] += int(has_stack)
            item["text_count"] += int(has_text)
            timestamp = (row.get("@timestamp") or "").strip()
            if not item["first_timestamp"] or timestamp < item["first_timestamp"]:
                item["first_timestamp"] = timestamp
                item["first_event_index"] = event_index
            if level == "ERROR" or has_stack or has_text:
                if not item["first_signal_index"]:
                    item["first_signal_index"] = event_index
            call_stack = row.get("tc_trace_call_stack") or ""
            item["callstack_hits"] += call_stack.count(f"):{service}/")

    variants = {
        "raw_evidence": lambda service, item: (-item["error_count"], -item["stack_count"],
            -item["text_count"], item["first_timestamp"], service),
        "volume_normalized": lambda service, item: (
            -(item["error_count"] / item["event_count"]),
            -(item["stack_count"] / item["event_count"]),
            -(item["text_count"] / item["event_count"]),
            -item["error_count"], service),
        "temporal_first_signal": lambda service, item: (
            item["first_signal_index"] or sys.maxsize, -item["error_count"],
            -item["stack_count"], service),
        "call_stack_evidence": lambda service, item: (
            -item["callstack_hits"], -item["error_count"], -item["stack_count"],
            item["first_signal_index"] or sys.maxsize, service),
    }
    top_by_variant = {}
    candidate_rows = []
    for trace_id, services in groups.items():
        top_by_variant[trace_id] = {}
        for variant, key_fn in variants.items():
            ordered = sorted(services.items(), key=lambda pair: key_fn(*pair))
            top_by_variant[trace_id][variant] = ordered[0][0]
            for rank, (service, item) in enumerate(ordered, start=1):
                candidate_rows.append({"tc_trace_id": trace_id, "variant": variant,
                    "rank": rank, "candidate_service": service, **item,
                    "global_service_events": global_service_events[service]})

    agreements = {}
    for variant in variants:
        if variant == "raw_evidence":
            continue
        agreements[variant] = sum(
            top_by_variant[trace][variant] == top_by_variant[trace]["raw_evidence"]
            for trace in top_by_variant
        )
    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tc_trace_id", "variant", "rank", "candidate_service", "event_count",
              "error_count", "stack_count", "text_count", "first_timestamp",
              "first_event_index", "first_signal_index", "callstack_hits",
              "global_service_events"]
    with args.candidates_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(candidate_rows)
    report = {"stage": 10, "stage_name": "compare_service_rankings",
              "alerted_traces": len(groups), "candidate_rows": len(candidate_rows),
              "variants": list(variants), "agreement_with_raw_top1": agreements,
              "labels_used": False, "root_cause_labels_available": False,
              "candidates_output": str(args.candidates_output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
