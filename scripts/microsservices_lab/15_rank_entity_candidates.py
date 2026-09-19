from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


LOGICAL_COMPONENTS = {"scheduler", "catalog", "security"}
CALLSTACK_RE = re.compile(r"\):([A-Za-z0-9_-]+)/")
TERM_RE = re.compile(r"\b(exception|failed|timeout|error|recusou|órfã|orf[aã])\b", re.I)


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
    parser.add_argument("--review", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage15_entity_ranking.csv"))
    parser.add_argument("--evidence-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage15_entity_evidence.csv"))
    args = parser.parse_args()

    alerts = {}
    with args.alerts.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("alert_level") != "none":
                alerts[row["tc_trace_id"].strip()] = row

    service_vocabulary: set[str] = set()
    with args.events.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            service = (row.get("tc_service") or "").strip()
            if service:
                service_vocabulary.add(service)

    groups: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "emitter_events": 0, "emitter_errors": 0, "mention_events": 0,
        "mention_errors": 0, "callstack_hits": 0, "first_signal_index": 0,
        "first_error_index": 0, "sources": set(),
    }))
    with args.events.open("r", encoding="utf-8-sig", newline="") as stream:
        for event_index, row in enumerate(csv.DictReader(stream), start=1):
            trace_id = (row.get("tc_trace_id") or "").strip()
            if trace_id not in alerts:
                continue
            emitter = (row.get("tc_service") or "").strip() or "<UNKNOWN_SERVICE>"
            level = (row.get("level") or "").strip().upper()
            is_error = level == "ERROR"
            text = " ".join((row.get(field) or "") for field in
                             ["message", "stack_trace", "logger_name", "tc_trace_call_stack"])
            callstack_services = set(CALLSTACK_RE.findall(row.get("tc_trace_call_stack") or ""))
            mentioned = {service for service in service_vocabulary if service in text}
            mentioned.update(component for component in LOGICAL_COMPONENTS
                             if re.search(rf"\b{re.escape(component)}\b", text, re.I))
            candidates = {emitter: {"emitter"}}
            for service in mentioned:
                candidates.setdefault(service, set()).add("mentioned")
            for service in callstack_services:
                candidates.setdefault(service, set()).add("callstack")
            for service, sources in candidates.items():
                item = groups[trace_id][service]
                item["sources"].update(sources)
                item["emitter_events"] += int(service == emitter)
                item["emitter_errors"] += int(service == emitter and is_error)
                item["mention_events"] += int(service in mentioned)
                item["mention_errors"] += int(service in mentioned and is_error)
                item["callstack_hits"] += int(service in callstack_services)
                if service in mentioned or is_error:
                    if not item["first_signal_index"]:
                        item["first_signal_index"] = event_index
                if is_error and not item["first_error_index"]:
                    item["first_error_index"] = event_index

    def ranking_key(pair: tuple[str, dict]) -> tuple:
        service, item = pair
        return (-item["mention_errors"], -item["callstack_hits"],
                -item["emitter_errors"], -item["mention_events"],
                item["first_signal_index"] or sys.maxsize, service)

    candidate_rows = []
    trace_rows = []
    for trace_id, services in groups.items():
        ordered = sorted(services.items(), key=ranking_key)
        for rank, (service, item) in enumerate(ordered, start=1):
            candidate_rows.append({"tc_trace_id": trace_id, **alerts[trace_id], "rank": rank,
                "candidate_entity": service, "entity_sources": "+".join(sorted(item["sources"])),
                **{key: value for key, value in item.items() if key != "sources"}})
        top = ordered[0]
        trace_rows.append({"tc_trace_id": trace_id, **alerts[trace_id],
                           "top_entity": top[0], "top_entity_sources": "+".join(sorted(top[1]["sources"]))})

    review = {}
    if args.review.exists():
        with args.review.open("r", encoding="utf-8-sig", newline="") as stream:
            review = {row["tc_trace_id"]: row for row in csv.DictReader(stream)}
    evaluated = [
        (trace_id, review[trace_id]["root_service_manual"].strip())
        for trace_id in review if trace_id in groups
        and review[trace_id].get("root_service_manual", "").strip().lower() not in {"", "nenhum"}
    ]
    top1 = sum(top_entity == root for trace_id, root in evaluated
               for top_entity in [trace_rows[[r["tc_trace_id"] for r in trace_rows].index(trace_id)]["top_entity"]])
    report = {"stage": 15, "stage_name": "rank_entity_candidates", "alerted_traces": len(groups),
              "candidate_rows": len(candidate_rows), "entity_sources": ["emitter", "mentioned", "callstack"],
              "logical_components": sorted(LOGICAL_COMPONENTS), "labels_used_for_ranking": False,
              "manual_evaluation": {"root_service_cases": len(evaluated), "top1_exact_match": top1,
                                     "top1_rate": top1 / len(evaluated) if evaluated else 0.0},
              "candidate_output": str(args.output), "trace_evidence_output": str(args.evidence_output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tc_trace_id", "anomaly_score", "alert_level", "manual_label", "rank",
              "candidate_entity", "entity_sources", "emitter_events", "emitter_errors",
              "mention_events", "mention_errors", "callstack_hits", "first_signal_index", "first_error_index"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(candidate_rows)
    with args.evidence_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(trace_rows)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
