from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage13_localization_evaluation.json"))
    parser.add_argument("--misses-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage13_localization_misses.csv"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    no_root_values = {"", "nenhum", "none", "nao_aplicavel", "não_aplicável"}
    filled = [row for row in rows if row.get("root_service_manual", "").strip().lower() not in no_root_values]
    top1 = [row for row in filled if row.get("root_candidate_rank") == "1"]
    top3 = [row for row in filled if row.get("root_in_top3", "").strip().lower() in {"sim", "yes", "true", "1"}]
    misses = [row for row in filled if row not in top3]

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})
    for row in filled:
        category = row.get("root_cause_category_manual", "").strip() or "<sem_categoria>"
        by_category[category]["total"] += 1
        by_category[category]["top1"] += int(row in top1)
        by_category[category]["top3"] += int(row in top3)
    category_metrics = {}
    for category, values in sorted(by_category.items()):
        category_metrics[category] = {**values,
            "top1_rate": values["top1"] / values["total"],
            "top3_rate": values["top3"] / values["total"]}

    args.misses_output.parent.mkdir(parents=True, exist_ok=True)
    miss_fields = ["tc_trace_id", "alert_level", "anomaly_score", "root_service_manual",
                   "root_cause_category_manual", "top_candidate_1", "top_candidate_2",
                   "top_candidate_3", "first_error_service", "entry_service", "evidence_manual"]
    with args.misses_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=miss_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(misses)

    report = {
        "stage": 13, "stage_name": "evaluate_service_localization",
        "input": str(args.input), "reviewed_traces": len(rows),
        "root_service_filled": len(filled), "root_service_missing": len(rows) - len(filled),
        "top1": {"correct": len(top1), "total": len(filled), "rate": len(top1) / len(filled) if filled else 0.0},
        "top3": {"correct": len(top3), "total": len(filled), "rate": len(top3) / len(filled) if filled else 0.0},
        "root_rank_distribution": dict(Counter(row.get("root_candidate_rank", "") for row in filled)),
        "by_root_cause_category": category_metrics,
        "misses_output": str(args.misses_output),
        "root_cause_labels_are_manual": True,
        "causal_ground_truth": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
