from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

configure_csv_limit()

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Ranqueia serviços candidatos nos traces alertados.")
    parser.add_argument("--alerts", type=Path, default=Path("data/processed/microservices_final/microservices_alerts.csv.gz"))
    parser.add_argument("--assignments", type=Path, default=Path("data/processed/microservices_final/microservices_events_drain3_assignments.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/microservices_final/microservices_service_candidates.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage09_service_localization.manifest.json"))
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Artefatos de localização já existem; remova-os deliberadamente para reexecutar.")
    alerts = pd.read_csv(args.alerts, compression="gzip", dtype=str)
    alerts = alerts[alerts["alert_level"].isin(["review", "high_confidence"])].copy()
    alert_map = dict(zip(alerts["analysis_trace_id"], alerts[["manual_label", "alert_level", "isolation_forest_score"]].to_dict("records")))
    service_stats = defaultdict(lambda: defaultdict(lambda: {"events": 0, "errors": 0, "warns": 0, "templates": set(), "first_ordinal": None}))
    scanned = 0
    with gzip.open(args.assignments, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            scanned += 1
            trace_id = row["analysis_trace_id"]
            if trace_id not in alert_map:
                continue
            service = (row.get("tc_service") or "").strip() or "<unknown_service>"
            stats = service_stats[trace_id][service]
            stats["events"] += 1
            level = (row.get("level") or "").strip().upper()
            stats["errors"] += int(level == "ERROR"); stats["warns"] += int(level == "WARN")
            stats["templates"].add(row.get("template_id") or "0")
            ordinal = int(row.get("event_ordinal") or 0)
            if stats["first_ordinal"] is None or ordinal < stats["first_ordinal"]:
                stats["first_ordinal"] = ordinal
            if scanned % 1_000_000 == 0:
                print(f"[localização] eventos lidos: {scanned:,}", flush=True)

    fields = ["analysis_trace_id", "alert_level", "anomaly_score", "manual_label", "candidate_rank", "tc_service", "service_events", "service_errors", "service_warns", "service_unique_templates", "first_event_ordinal", "ranking_basis"]
    rows = []
    summary = {"traces_alerted": len(alert_map), "traces_with_candidates": 0, "top1_candidates": 0, "top3_candidates": 0}
    for trace_id, candidates in service_stats.items():
        ranked = sorted(candidates.items(), key=lambda item: (-item[1]["errors"], -len(item[1]["templates"]), -item[1]["events"], item[1]["first_ordinal"], item[0]))
        if ranked:
            summary["traces_with_candidates"] += 1
        for rank, (service, stats) in enumerate(ranked, start=1):
            if rank <= 3:
                summary["top3_candidates"] += 1
            if rank == 1:
                summary["top1_candidates"] += 1
            rows.append({"analysis_trace_id": trace_id, "alert_level": alert_map[trace_id]["alert_level"], "anomaly_score": alert_map[trace_id]["isolation_forest_score"], "manual_label": alert_map[trace_id]["manual_label"], "candidate_rank": rank, "tc_service": service, "service_events": stats["events"], "service_errors": stats["errors"], "service_warns": stats["warns"], "service_unique_templates": len(stats["templates"]), "first_event_ordinal": stats["first_ordinal"], "ranking_basis": "error_count desc, unique_templates desc, events desc, first_ordinal asc"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, args.output)
    manifest = {"stage": 9, "stage_name": "localize_candidate_services", "generated_at": datetime.now(timezone.utc).isoformat(), "inputs": {"alerts": {"path": str(args.alerts), "sha256": sha256_file(args.alerts)}, "assignments": {"path": str(args.assignments), "sha256": sha256_file(args.assignments)}}, "policy": {"candidate_unit": "tc_service", "ranking": "error_count desc, unique_templates desc, events desc, first_event_ordinal asc", "interpretation": "candidate_service_with_evidence_not_proof"}, "results": {"events_scanned": scanned, **summary, "candidate_rows": len(rows)}, "output": {"path": str(args.output), "sha256": sha256_file(args.output)}, "checks": {"alerted_traces_nonzero": bool(alert_map), "candidates_nonzero": bool(rows), "output_hash_recorded": True}}
    manifest["accepted"] = all(manifest["checks"].values())
    args.manifest.parent.mkdir(parents=True, exist_ok=True); temporary_manifest = Path(str(args.manifest) + ".tmp"); temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
