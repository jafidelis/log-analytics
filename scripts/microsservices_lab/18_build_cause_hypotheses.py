from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SCHEDULER_RE = re.compile(r"scheduler\s+recusou|task[^\n]{0,160}(?:órfã|orf[aã]|em\s+execução|em\s+execucao)", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_evidence.csv"))
    parser.add_argument("--review", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage18_cause_hypotheses.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage18_cause_hypotheses.manifest.json"))
    args = parser.parse_args()

    review = {}
    with args.review.open(encoding="utf-8-sig", newline="") as stream:
        review = {r["tc_trace_id"]: r for r in csv.DictReader(stream)}
    output_rows = []
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            trace_id = row["tc_trace_id"]
            text = " ".join(row.get(field, "") for field in
                             ["first_error_message", "first_exception", "top_warn_patterns"])
            explicit_scheduler = bool(SCHEDULER_RE.search(text))
            if explicit_scheduler:
                hypothesis = "scheduler"
                source = "explicit_logical_component"
                confidence = "high_relative"
            elif row.get("first_error_service", "").strip():
                hypothesis = row["first_error_service"].strip()
                source = "first_error_service"
                confidence = "medium_relative"
            else:
                hypothesis = row.get("entry_service", "").strip() or "<UNKNOWN_SERVICE>"
                source = "entry_service"
                confidence = "low_relative"
            manual = review.get(trace_id, {})
            output_rows.append({"tc_trace_id": trace_id, "hypothesis_entity": hypothesis,
                "hypothesis_source": source, "relative_confidence": confidence,
                "explicit_scheduler_evidence": int(explicit_scheduler),
                "first_error_service": row.get("first_error_service", ""),
                "entry_service": row.get("entry_service", ""),
                "root_service_manual": manual.get("root_service_manual", ""),
                "root_cause_category_manual": manual.get("root_cause_category_manual", "")})

    evaluated = [r for r in output_rows if r["root_service_manual"].strip().lower() not in {"", "nenhum"}]
    exact = sum(r["hypothesis_entity"].lower() == r["root_service_manual"].strip().lower() for r in evaluated)
    report = {"stage": 18, "stage_name": "build_cause_hypotheses", "traces": len(output_rows),
              "hypothesis_distribution": {}, "manual_evaluation": {"cases": len(evaluated),
                  "exact_match": exact, "exact_match_rate": exact / len(evaluated) if evaluated else 0.0},
              "labels_used_for_hypothesis": False, "causal_ground_truth": False,
              "output": str(args.output)}
    for row in output_rows:
        key = row["hypothesis_source"]
        report["hypothesis_distribution"][key] = report["hypothesis_distribution"].get(key, 0) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
