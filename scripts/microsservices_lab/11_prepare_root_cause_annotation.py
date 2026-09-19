from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def evenly_select(rows: list[dict[str, str]], amount: int) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda row: (-float(row["anomaly_score"]), row["tc_trace_id"]))
    if len(rows) <= amount:
        return rows
    positions = [round(i * (len(rows) - 1) / (amount - 1)) for i in range(amount)]
    return [rows[position] for position in positions]


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts", type=Path, default=Path(
        "artifacts/microsservices_lab/stage8_trace_alerts.csv"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "artifacts/microsservices_lab/stage10_service_ranking_candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage11_root_cause_annotation.csv"))
    parser.add_argument("--per-level", type=int, default=25)
    args = parser.parse_args()

    alerts = []
    with args.alerts.open("r", encoding="utf-8-sig", newline="") as stream:
        alerts = [row for row in csv.DictReader(stream) if row["alert_level"] != "none"]
    by_level = defaultdict(list)
    for row in alerts:
        by_level[row["alert_level"]].append(row)
    selected = []
    for level in ["high_confidence", "review"]:
        selected.extend(evenly_select(by_level[level], args.per_level))

    candidates = defaultdict(list)
    with args.candidates.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["rank"]) <= 3:
                candidates[row["tc_trace_id"]].append(row)

    output_rows = []
    for row in sorted(selected, key=lambda item: (-float(item["anomaly_score"]), item["tc_trace_id"])):
        top = sorted(candidates[row["tc_trace_id"]], key=lambda item: int(item["rank"]))
        candidate_names = [item["candidate_service"] for item in top]
        output_rows.append({
            "tc_trace_id": row["tc_trace_id"],
            "anomaly_score": row["anomaly_score"],
            "alert_level": row["alert_level"],
            "event_count": row["event_count"],
            "service_count": row["service_count"],
            "top_candidate_1": candidate_names[0] if len(candidate_names) > 0 else "",
            "top_candidate_2": candidate_names[1] if len(candidate_names) > 1 else "",
            "top_candidate_3": candidate_names[2] if len(candidate_names) > 2 else "",
            "root_service_manual": "",
            "root_cause_category_manual": "",
            "evidence_manual": "",
            "reviewer_notes": "",
        })

    fields = list(output_rows[0]) if output_rows else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    report = {"stage": 11, "stage_name": "prepare_root_cause_annotation",
              "selected_traces": len(output_rows),
              "selection": {"per_alert_level": args.per_level, "levels": sorted(by_level)},
              "labels_in_annotation_sheet": False,
              "manual_fields": ["root_service_manual", "root_cause_category_manual", "evidence_manual", "reviewer_notes"],
              "output": str(args.output)}
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
