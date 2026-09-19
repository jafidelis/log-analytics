from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "artifacts/microsservices_lab/stage22_structured_cause_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage23_structured_review_quality.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage23_structured_review_quality.manifest.json"))
    args = parser.parse_args()
    required = ["observed_fact_automatic", "interpretation_manual", "cause_hypothesis_manual",
                "evidence_reference_manual", "confidence_manual", "alternative_hypothesis_manual"]
    output = []
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            checks = {field: bool(row.get(field, "").strip()) for field in required}
            score = sum(checks.values())
            output.append({"tc_trace_id": row["tc_trace_id"], **{f"filled_{field}": int(value) for field, value in checks.items()},
                           "structured_completeness_score": score, "structured_completeness_max": len(required),
                           "confidence_value": row.get("confidence_manual", "").strip().lower()})
    report = {"stage": 23, "stage_name": "score_structured_review_quality", "reviewed_traces": len(output),
              "rubric": {field: "1 point" for field in required},
              "score_distribution": dict(sorted(Counter(str(row["structured_completeness_score"]) for row in output).items())),
              "mean_score": sum(row["structured_completeness_score"] for row in output) / len(output) if output else 0.0,
              "complete_reviews": sum(row["structured_completeness_score"] == len(required) for row in output),
              "confidence_distribution": dict(Counter(row["confidence_value"] for row in output)),
              "causal_correctness_validated": False, "manual_annotations_used": True,
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
