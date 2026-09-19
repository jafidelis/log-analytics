from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


SCHEDULER_RE = re.compile(
    r"(?:scheduler\s+recusou|recusou\s+\d+\s+de\s+\d+\s+task|task[^\n]{0,120}(?:órfã|orf[aã]|em\s+execução|em\s+execucao))",
    re.I,
)


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
    parser.add_argument("--candidates", type=Path, default=Path(
        "artifacts/microsservices_lab/stage16_normalized_entity_ranking.csv"))
    parser.add_argument("--review", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage17_scheduler_rule_ranking.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage17_scheduler_rule.manifest.json"))
    args = parser.parse_args()

    alert_ids = set()
    with args.alerts.open(encoding="utf-8-sig", newline="") as stream:
        alert_ids = {r["tc_trace_id"].strip() for r in csv.DictReader(stream) if r["alert_level"] != "none"}
    scheduler_evidence: dict[str, dict[str, int]] = defaultdict(lambda: {"matches": 0, "first_event": 0})
    with args.events.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=1):
            trace_id = (row.get("tc_trace_id") or "").strip()
            if trace_id not in alert_ids:
                continue
            text = " ".join((row.get(field) or "") for field in ["message", "stack_trace", "logger_name", "tc_trace_call_stack"])
            if SCHEDULER_RE.search(text):
                evidence = scheduler_evidence[trace_id]
                evidence["matches"] += 1
                if not evidence["first_event"]:
                    evidence["first_event"] = index

    groups: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "mention_errors": 0, "callstack_hits": 0, "emitter_errors": 0,
        "mention_events": 0, "first_signal_index": 0, "first_error_index": 0,
        "explicit_scheduler_matches": 0, "source": "",
    }))
    with args.candidates.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            trace_id = row["tc_trace_id"]
            service = row["candidate_entity"]
            item = groups[trace_id][service]
            for key in ["mention_errors", "callstack_hits", "emitter_errors", "mention_events"]:
                item[key] += int(row[key])
            for key in ["first_signal_index", "first_error_index"]:
                value = int(row[key])
                if value and (not item[key] or value < item[key]):
                    item[key] = value
            item["source"] = row.get("entity_sources", "")
    for trace_id, evidence in scheduler_evidence.items():
        if evidence["matches"]:
            item = groups[trace_id]["scheduler"]
            item["explicit_scheduler_matches"] += evidence["matches"]
            item["mention_errors"] += evidence["matches"]
            item["mention_events"] += evidence["matches"]
            item["first_signal_index"] = evidence["first_event"]
            item["source"] = "+".join(filter(None, [item["source"], "explicit_scheduler_rule"]))

    top = {}
    output_rows = []
    for trace_id, services in groups.items():
        ordered = sorted(services.items(), key=lambda pair: (
            -pair[1]["explicit_scheduler_matches"], -pair[1]["mention_errors"],
            -pair[1]["callstack_hits"], -pair[1]["emitter_errors"],
            -pair[1]["mention_events"], pair[1]["first_signal_index"] or sys.maxsize, pair[0]))
        top[trace_id] = [service for service, _ in ordered]
        for rank, (service, item) in enumerate(ordered, start=1):
            output_rows.append({"tc_trace_id": trace_id, "rank": rank, "candidate_entity": service,
                "entity_sources": item["source"], **{key: value for key, value in item.items() if key != "source"}})

    manual = {}
    with args.review.open(encoding="utf-8-sig", newline="") as stream:
        manual = {r["tc_trace_id"]: r for r in csv.DictReader(stream)}
    evaluated = []
    for trace_id, row in manual.items():
        root = row.get("root_service_manual", "").strip().lower()
        if root and root != "nenhum" and trace_id in top:
            evaluated.append({"root": root, "top1": top[trace_id][0] == root, "top3": root in top[trace_id][:3]})
    report = {"stage": 17, "stage_name": "scheduler_specific_rule", "alerted_traces": len(groups),
              "scheduler_evidence_traces": len(scheduler_evidence),
              "scheduler_evidence_matches": sum(x["matches"] for x in scheduler_evidence.values()),
              "manual_evaluation": {"cases": len(evaluated), "top1": sum(x["top1"] for x in evaluated),
                  "top3": sum(x["top3"] for x in evaluated),
                  "top1_rate": sum(x["top1"] for x in evaluated) / len(evaluated) if evaluated else 0.0,
                  "top3_rate": sum(x["top3"] for x in evaluated) / len(evaluated) if evaluated else 0.0},
              "labels_used_for_rule": False, "rule": SCHEDULER_RE.pattern,
              "output": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tc_trace_id", "rank", "candidate_entity", "entity_sources", "mention_errors",
              "callstack_hits", "emitter_errors", "mention_events", "first_signal_index",
              "first_error_index", "explicit_scheduler_matches"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
