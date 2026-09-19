from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


LOGICAL = {"scheduler", "catalog", "security"}


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def canonicalize(name: str, vocabulary: set[str]) -> str | None:
    name = name.strip().lower()
    if name in vocabulary or name in LOGICAL:
        return name
    matches = [service for service in vocabulary if name.endswith(f"-{service}")]
    if len(matches) == 1:
        return matches[0]
    return None


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path(
        "artifacts/microsservices_lab/stage15_entity_ranking.csv"))
    parser.add_argument("--events", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_traces_anonymized.csv"))
    parser.add_argument("--review", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage16_normalized_entity_ranking.csv"))
    parser.add_argument("--evidence-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage16_normalized_entity_evidence.csv"))
    args = parser.parse_args()

    vocabulary = set()
    with args.events.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("tc_service", "").strip():
                vocabulary.add(row["tc_service"].strip().lower())

    groups: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "emitter_events": 0, "emitter_errors": 0, "mention_events": 0,
        "mention_errors": 0, "callstack_hits": 0, "first_signal_index": 0,
        "first_error_index": 0, "sources": set(),
    }))
    with args.candidates.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            canonical = canonicalize(row["candidate_entity"], vocabulary)
            if canonical is None:
                continue
            item = groups[row["tc_trace_id"]][canonical]
            for key in ["emitter_events", "emitter_errors", "mention_events", "mention_errors",
                        "callstack_hits", "first_signal_index", "first_error_index"]:
                value = int(row[key])
                if key.startswith("first_"):
                    if value and (not item[key] or value < item[key]):
                        item[key] = value
                else:
                    item[key] += value
            item["sources"].update(filter(None, row.get("entity_sources", "").split("+")))

    def rank_key(pair: tuple[str, dict]) -> tuple:
        service, item = pair
        return (-item["mention_errors"], -item["callstack_hits"], -item["emitter_errors"],
                -item["mention_events"], item["first_signal_index"] or 10**18, service)

    candidate_rows = []
    trace_rows = []
    top_by_trace = {}
    for trace_id, services in groups.items():
        ordered = sorted(services.items(), key=rank_key)
        top_by_trace[trace_id] = [service for service, _ in ordered]
        for rank, (service, item) in enumerate(ordered, start=1):
            candidate_rows.append({"tc_trace_id": trace_id, "rank": rank,
                "candidate_entity": service, "entity_sources": "+".join(sorted(item["sources"])),
                **{key: value for key, value in item.items() if key != "sources"}})
        if ordered:
            trace_rows.append({"tc_trace_id": trace_id, "top_entity": ordered[0][0],
                               "top3_entities": "|".join(service for service, _ in ordered[:3])})

    review = {}
    with args.review.open("r", encoding="utf-8-sig", newline="") as stream:
        review = {row["tc_trace_id"]: row for row in csv.DictReader(stream)}
    evaluated = []
    for trace_id, row in review.items():
        root = row.get("root_service_manual", "").strip().lower()
        if root and root != "nenhum" and trace_id in top_by_trace:
            ranked = top_by_trace[trace_id]
            evaluated.append({"tc_trace_id": trace_id, "root": root,
                              "top1": ranked[0] == root,
                              "top3": root in ranked[:3]})
    report = {"stage": 16, "stage_name": "normalize_entity_ranking",
              "alerted_traces": len(groups), "candidate_rows": len(candidate_rows),
              "service_vocabulary_size": len(vocabulary), "logical_components": sorted(LOGICAL),
              "discarded_generic_or_unknown_entities": True,
              "manual_evaluation": {"cases": len(evaluated),
                  "top1": sum(item["top1"] for item in evaluated),
                  "top3": sum(item["top3"] for item in evaluated),
                  "top1_rate": sum(item["top1"] for item in evaluated) / len(evaluated) if evaluated else 0.0,
                  "top3_rate": sum(item["top3"] for item in evaluated) / len(evaluated) if evaluated else 0.0},
              "labels_used_for_ranking": False,
              "candidate_output": str(args.output), "evidence_output": str(args.evidence_output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tc_trace_id", "rank", "candidate_entity", "entity_sources", "emitter_events",
              "emitter_errors", "mention_events", "mention_errors", "callstack_hits",
              "first_signal_index", "first_error_index"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(candidate_rows)
    with args.evidence_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(trace_rows)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
