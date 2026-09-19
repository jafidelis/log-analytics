from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b")


def read_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["tc_trace_id"]: row for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", type=Path, default=Path(
        "artifacts/microsservices_lab/stage22_structured_cause_review.csv"))
    parser.add_argument("--manual", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--hypotheses", type=Path, default=Path(
        "artifacts/microsservices_lab/stage19_explainable_cause_evidence.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage24_explainable_output_evaluation.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage24_explainable_output_evaluation.manifest.json"))
    args = parser.parse_args()

    manual = read_map(args.manual)
    hypotheses = read_map(args.hypotheses)
    output = []
    with args.structured.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            trace_id = row["tc_trace_id"]
            m = manual.get(trace_id, {})
            h = hypotheses.get(trace_id, {})
            evidence = row.get("evidence_reference_manual", "")
            root = m.get("root_service_manual", "").strip().lower()
            rank = m.get("root_candidate_rank", "").strip()
            top3 = m.get("root_in_top3", "").strip().lower() == "sim"
            confidence = row.get("confidence_manual", "").strip().lower()
            output.append({
                "tc_trace_id": trace_id, "confidence": confidence,
                "root_service_manual": root, "root_cause_category_manual": m.get("root_cause_category_manual", ""),
                "hypothesis": h.get("hypothesis", ""), "hypothesis_source": h.get("evidence_type", ""),
                "root_candidate_rank": rank, "root_in_top3": int(top3),
                "hypothesis_matches_root": int(bool(root and root != "nenhum" and h.get("hypothesis", "").strip().lower() == root)),
                "evidence_has_timestamp": int(bool(TIME_RE.search(evidence))),
                "evidence_has_root_service": int(bool(root and root != "nenhum" and root in evidence.lower())),
                "evidence_has_event_reference": int(bool(evidence.strip())),
                "evidence_has_causal_reasoning": int(bool(row.get("interpretation_manual", "").strip() and row.get("cause_hypothesis_manual", "").strip())),
            })
    evaluated = [r for r in output if r["root_service_manual"] not in {"", "nenhum"}]
    for row in evaluated:
        row["traceability_score"] = sum(row[field] for field in ["evidence_has_timestamp", "evidence_has_root_service", "evidence_has_event_reference"])
    by_confidence = defaultdict(list)
    for row in evaluated:
        by_confidence[row["confidence"]].append(row)
    confidence_summary = {}
    for confidence, rows in sorted(by_confidence.items()):
        confidence_summary[confidence] = {"n": len(rows),
            "top1_rate": sum(r["root_candidate_rank"] == "1" for r in rows) / len(rows),
            "top3_rate": sum(r["root_in_top3"] for r in rows) / len(rows),
            "hypothesis_match_rate": sum(r["hypothesis_matches_root"] for r in rows) / len(rows),
            "mean_traceability_score": sum(r["traceability_score"] for r in rows) / len(rows)}
    report = {"stage": 24, "stage_name": "evaluate_explainable_outputs", "reviewed_traces": len(output),
              "root_service_cases": len(evaluated), "confidence_summary": confidence_summary,
              "overall": {"top1_rate": sum(r["root_candidate_rank"] == "1" for r in evaluated) / len(evaluated) if evaluated else 0.0,
                          "top3_rate": sum(r["root_in_top3"] for r in evaluated) / len(evaluated) if evaluated else 0.0,
                          "hypothesis_match_rate": sum(r["hypothesis_matches_root"] for r in evaluated) / len(evaluated) if evaluated else 0.0,
                          "mean_traceability_score": sum(r["traceability_score"] for r in evaluated) / len(evaluated) if evaluated else 0.0},
              "causal_correctness_validated": False, "manual_annotations_used": True,
              "output": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0])
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
