from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b")
SIGNAL_RE = re.compile(r"\b(ERROR|WARN|Exception|timeout|failed|recusou|órfã|orf[aã])\b", re.I)
RATIONALE_RE = re.compile(r"\b(causa|nasce|propaga|devido|porque|consistente|precedido|origem|responsável|responsavel|falha)\b", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage21_justification_quality.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage21_justification_quality.manifest.json"))
    args = parser.parse_args()

    output = []
    with args.review.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            service = row.get("root_service_manual", "").strip()
            evidence = row.get("evidence_manual", "").strip()
            category = row.get("root_cause_category_manual", "").strip()
            service_present = bool(service and service.lower() != "nenhum" and service.lower() in evidence.lower())
            time_present = bool(TIME_RE.search(evidence))
            signal_present = bool(SIGNAL_RE.search(evidence))
            rationale_present = bool(RATIONALE_RE.search(evidence))
            score = sum([bool(evidence), service_present, time_present, signal_present, rationale_present])
            output.append({"tc_trace_id": row["tc_trace_id"], "root_cause_category_manual": category,
                "evidence_text_present": int(bool(evidence)), "root_service_in_evidence": int(service_present),
                "timestamp_in_evidence": int(time_present), "signal_in_evidence": int(signal_present),
                "rationale_in_evidence": int(rationale_present), "justification_completeness_score": score,
                "justification_max_score": 5})
    distribution = Counter(str(row["justification_completeness_score"]) for row in output)
    report = {"stage": 21, "stage_name": "score_causal_justifications", "reviewed_traces": len(output),
              "rubric": {"evidence_text": "1 point", "root_service": "1 point", "timestamp": "1 point",
                          "signal_or_event": "1 point", "causal_rationale": "1 point"},
              "score_distribution": dict(sorted(distribution.items())),
              "mean_score": sum(row["justification_completeness_score"] for row in output) / len(output) if output else 0.0,
              "complete_justifications": sum(row["justification_completeness_score"] == 5 for row in output),
              "causal_correctness_validated": False, "manual_annotations_used_for_rubric": True,
              "anomaly_labels_used_for_scoring": False,
              "output": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
