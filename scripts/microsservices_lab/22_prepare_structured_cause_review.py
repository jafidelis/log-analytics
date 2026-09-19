from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_evidence.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage22_structured_cause_review.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "artifacts/microsservices_lab/stage22_structured_cause_review.manifest.json"))
    args = parser.parse_args()

    rows = []
    with args.evidence.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            observed = []
            if row.get("entry_service", "").strip():
                observed.append(f"entrada={row['entry_service'].strip()}")
            if row.get("first_error_service", "").strip():
                observed.append(f"primeiro_servico_com_erro={row['first_error_service'].strip()}")
            if row.get("first_error_timestamp", "").strip():
                observed.append(f"primeiro_timestamp_com_erro={row['first_error_timestamp'].strip()}")
            if row.get("error_services", "").strip():
                observed.append(f"servicos_com_erro={row['error_services'].strip()}")
            rows.append({
                "tc_trace_id": row["tc_trace_id"],
                "anomaly_score": row.get("anomaly_score", ""),
                "alert_level": row.get("alert_level", ""),
                "observed_fact_automatic": " | ".join(observed),
                "first_error_message_reference": row.get("first_error_message", ""),
                "top_warn_patterns_reference": row.get("top_warn_patterns", ""),
                "interpretation_manual": "",
                "cause_hypothesis_manual": "",
                "evidence_reference_manual": "",
                "confidence_manual": "",
                "alternative_hypothesis_manual": "",
                "reviewer_notes": "",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {"stage": 22, "stage_name": "prepare_structured_cause_review",
              "traces": len(rows), "automatic_facts_prefilled": True,
              "manual_fields": ["interpretation_manual", "cause_hypothesis_manual",
                  "evidence_reference_manual", "confidence_manual", "alternative_hypothesis_manual", "reviewer_notes"],
              "causal_ground_truth": False, "output": str(args.output)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
