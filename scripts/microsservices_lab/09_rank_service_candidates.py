from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


TERMS = re.compile(r"\b(exception|failed|timeout|error)\b", re.I)


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
        "artifacts/microsservices_lab/stage9_service_candidates.csv"))
    parser.add_argument("--traces-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage9_trace_root_cause_evidence.csv"))
    args = parser.parse_args()

    alert_ids: dict[str, dict[str, str]] = {}
    with args.alerts.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("alert_level") != "none":
                alert_ids[row["tc_trace_id"].strip()] = row
    if not alert_ids:
        raise ValueError("Nenhum trace alertado foi encontrado.")

    groups: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "event_count": 0, "error_count": 0, "warn_count": 0,
        "stack_trace_count": 0, "text_signal_event_count": 0,
        "first_timestamp": "", "first_event_index": 0,
    }))
    event_index = 0
    with args.events.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            event_index += 1
            trace_id = (row.get("tc_trace_id") or "").strip()
            if trace_id not in alert_ids:
                continue
            service = (row.get("tc_service") or "").strip() or "<UNKNOWN_SERVICE>"
            item = groups[trace_id][service]
            item["event_count"] += 1
            level = (row.get("level") or "").strip().upper()
            item["error_count"] += int(level == "ERROR")
            item["warn_count"] += int(level == "WARN")
            item["stack_trace_count"] += int(bool((row.get("stack_trace") or "").strip()))
            item["text_signal_event_count"] += int(bool(TERMS.search(
                f"{row.get('message') or ''} {row.get('stack_trace') or ''}")))
            timestamp = (row.get("@timestamp") or "").strip()
            if not item["first_timestamp"] or timestamp < item["first_timestamp"]:
                item["first_timestamp"] = timestamp
                item["first_event_index"] = event_index

    candidate_rows = []
    evidence_rows = []
    for trace_id, services in groups.items():
        alert_context = {
            "anomaly_score": alert_ids[trace_id]["anomaly_score"],
            "alert_level": alert_ids[trace_id]["alert_level"],
            "manual_label": alert_ids[trace_id].get("manual_label", ""),
        }
        ordered = sorted(services.items(), key=lambda pair: (
            -pair[1]["error_count"], -pair[1]["stack_trace_count"],
            -pair[1]["text_signal_event_count"], pair[1]["first_timestamp"], pair[0]))
        for rank, (service, item) in enumerate(ordered, start=1):
            candidate_rows.append({"tc_trace_id": trace_id, **alert_context,
                "service_rank": rank, "candidate_service": service, **item})
        top = ordered[0]
        evidence_rows.append({"tc_trace_id": trace_id, **alert_context,
            "top_service": top[0], "top_service_rank": 1,
            "candidate_service_count": len(ordered),
            "top_error_count": top[1]["error_count"],
            "top_stack_trace_count": top[1]["stack_trace_count"],
            "top_text_signal_event_count": top[1]["text_signal_event_count"],
            "top_first_timestamp": top[1]["first_timestamp"]})

    candidate_fields = ["tc_trace_id", "anomaly_score", "alert_level", "manual_label",
                        "service_rank", "candidate_service", "event_count", "error_count",
                        "warn_count", "stack_trace_count", "text_signal_event_count",
                        "first_timestamp", "first_event_index"]
    evidence_fields = ["tc_trace_id", "anomaly_score", "alert_level", "manual_label",
                       "top_service", "top_service_rank", "candidate_service_count",
                       "top_error_count", "top_stack_trace_count",
                       "top_text_signal_event_count", "top_first_timestamp"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields)
        writer.writeheader(); writer.writerows(candidate_rows)
    with args.traces_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=evidence_fields)
        writer.writeheader(); writer.writerows(evidence_rows)
    report = {"stage": 9, "stage_name": "rank_service_candidates",
              "alerted_traces": len(alert_ids), "candidate_rows": len(candidate_rows),
              "ranking": ["error_count desc", "stack_trace_count desc",
                          "text_signal_event_count desc", "first_timestamp asc", "service asc"],
              "labels_used_for_ranking": False, "root_cause_labels_available": False,
              "candidate_output": str(args.output), "trace_evidence_output": str(args.traces_output)}
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
