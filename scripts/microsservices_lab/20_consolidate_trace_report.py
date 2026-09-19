from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["tc_trace_id"]: row for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts", type=Path, default=Path(
        "artifacts/microsservices_lab/stage8_trace_alerts.csv"))
    parser.add_argument("--service-evidence", type=Path, default=Path(
        "artifacts/microsservices_lab/stage9_trace_root_cause_evidence.csv"))
    parser.add_argument("--cause-evidence", type=Path, default=Path(
        "artifacts/microsservices_lab/stage19_explainable_cause_evidence.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage20_trace_report.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage20_trace_report.manifest.json"))
    args = parser.parse_args()

    alerts = read_map(args.alerts)
    services = read_map(args.service_evidence)
    causes = read_map(args.cause_evidence)
    rows = []
    for trace_id, alert in alerts.items():
        service = services.get(trace_id, {})
        cause = causes.get(trace_id, {})
        rows.append({
            "tc_trace_id": trace_id,
            "anomaly_score": alert.get("anomaly_score", ""),
            "alert_level": alert.get("alert_level", ""),
            "detector_decision": "alert" if alert.get("alert_level") != "none" else "none",
            "top_service_candidate": service.get("top_service", ""),
            "service_candidate_count": service.get("candidate_service_count", ""),
            "cause_hypothesis": cause.get("hypothesis", ""),
            "hypothesis_source": cause.get("evidence_type", ""),
            "candidate_entities": cause.get("candidate_entities", ""),
            "first_error_service": cause.get("first_error_service", "") or service.get("top_service", ""),
            "entry_service": cause.get("entry_service", ""),
            "root_cause_category_manual": cause.get("root_cause_category_manual", ""),
            "evidence_entity_present": int(bool(cause.get("hypothesis", "").strip() or service.get("top_service", "").strip())),
            "evidence_service_present": int(bool(cause.get("first_error_service", "").strip() or cause.get("entry_service", "").strip())),
            "evidence_timestamp_present": int(bool(service.get("top_first_timestamp", "").strip())),
            "evidence_event_present": int(bool(cause.get("evidence_type", "").strip())),
            "manual_category_present": int(bool(cause.get("root_cause_category_manual", "").strip())),
        })

    reviewed = [row for row in rows if row["manual_category_present"]]
    coverage = {}
    for field in ["evidence_entity_present", "evidence_service_present", "evidence_timestamp_present", "evidence_event_present", "manual_category_present"]:
        coverage[field] = {"present": sum(row[field] for row in reviewed), "total": len(reviewed),
                           "rate": sum(row[field] for row in reviewed) / len(reviewed) if reviewed else 0.0}
    report = {"stage": 20, "stage_name": "consolidate_trace_report", "traces": len(rows),
              "alerted_traces": sum(row["detector_decision"] == "alert" for row in rows),
              "reviewed_traces_with_manual_category": len(reviewed),
              "justification_evidence_coverage": coverage,
              "labels_used_for_consolidation": False,
              "causal_ground_truth": False,
              "output": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
