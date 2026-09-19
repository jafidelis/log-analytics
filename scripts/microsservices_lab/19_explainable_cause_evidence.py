from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SCHEDULER_RE = re.compile(r"scheduler\s+recusou|task[^\n]{0,160}(?:órfã|orf[aã]|em\s+execução|em\s+execucao)", re.I)
TARGET_RE = re.compile(r"service:\s*([A-Za-z0-9_-]+)", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_evidence.csv"))
    parser.add_argument("--review", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage19_explainable_cause_evidence.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage19_explainable_cause_evidence.manifest.json"))
    args = parser.parse_args()

    manual = {}
    with args.review.open(encoding="utf-8-sig", newline="") as stream:
        manual = {r["tc_trace_id"]: r for r in csv.DictReader(stream)}
    output = []
    with args.evidence.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            text = " ".join(row.get(k, "") for k in ["first_error_message", "first_exception", "top_warn_patterns"])
            scheduler = bool(SCHEDULER_RE.search(text))
            targets = sorted(set(TARGET_RE.findall(text)), key=str.lower)
            error_services = [part.split(":", 1)[0] for part in row.get("error_services", "").split("|") if ":" in part]
            candidates = []
            if scheduler:
                candidates.append("scheduler")
            candidates.extend(targets)
            candidates.extend(error_services)
            if row.get("first_error_service", "").strip():
                candidates.append(row["first_error_service"].strip())
            if row.get("entry_service", "").strip():
                candidates.append(row["entry_service"].strip())
            candidates = list(dict.fromkeys(candidates))
            if len(error_services) > 1 and scheduler:
                hypothesis = "MULTI_HYPOTHESIS"
                evidence_type = "logical_component_plus_multiple_error_services"
            elif not error_services and targets:
                hypothesis = targets[0]
                evidence_type = "explicit_service_target_without_error"
            elif scheduler:
                hypothesis = "scheduler"
                evidence_type = "explicit_logical_component"
            elif row.get("first_error_service", "").strip():
                hypothesis = row["first_error_service"].strip()
                evidence_type = "first_error_service"
            else:
                hypothesis = row.get("entry_service", "").strip() or "<UNKNOWN_SERVICE>"
                evidence_type = "entry_service"
            m = manual.get(row["tc_trace_id"], {})
            output.append({"tc_trace_id": row["tc_trace_id"], "anomaly_score": row.get("anomaly_score", ""),
                "hypothesis": hypothesis, "evidence_type": evidence_type,
                "candidate_entities": "|".join(candidates), "scheduler_evidence": int(scheduler),
                "target_entities": "|".join(targets), "error_services": "|".join(error_services),
                "first_error_service": row.get("first_error_service", ""),
                "entry_service": row.get("entry_service", ""),
                "root_service_manual": m.get("root_service_manual", ""),
                "root_cause_category_manual": m.get("root_cause_category_manual", "")})
    evaluated = [r for r in output if r["root_service_manual"].strip().lower() not in {"", "nenhum"}]
    exact = sum(r["hypothesis"].lower() == r["root_service_manual"].strip().lower() for r in evaluated)
    report = {"stage": 19, "stage_name": "explainable_cause_evidence", "traces": len(output),
              "hypothesis_types": dict(Counter(r["evidence_type"] for r in output)),
              "manual_evaluation": {"cases": len(evaluated), "exact_match": exact,
                  "exact_match_rate": exact / len(evaluated) if evaluated else 0.0},
              "labels_used_for_hypothesis": False, "causal_ground_truth": False, "output": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
