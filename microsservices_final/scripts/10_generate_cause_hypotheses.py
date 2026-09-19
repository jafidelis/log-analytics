from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPLICIT_PATTERNS = [
    re.compile(r"\bserviceStarted\s*[:=]\s*([A-Za-z0-9_.-]+)", re.I),
    re.compile(r"\b(?:serviço|service)\s+(?:responsável|responsible|target|alvo)\s*[:=]?\s*([A-Za-z0-9_.-]+)", re.I),
]

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


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser(description="Gera hipóteses causais rastreáveis.")
    parser.add_argument("--alerts", type=Path, default=Path("data/processed/microservices_final/microservices_alerts.csv.gz"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/microservices_final/microservices_service_candidates.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_traceable.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/microservices_final/microservices_cause_hypotheses.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage10_cause_hypotheses.manifest.json"))
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Artefatos de hipóteses já existem; remova-os deliberadamente para reexecutar.")
    alerts = pd.read_csv(args.alerts, compression="gzip", dtype=str)
    alerts = alerts[alerts["alert_level"].isin(["review", "high_confidence"])].copy()
    alert_map = alerts.set_index("analysis_trace_id")["alert_level"].to_dict()
    label_map = alerts.set_index("analysis_trace_id")["manual_label"].to_dict()
    score_map = alerts.set_index("analysis_trace_id")["isolation_forest_score"].to_dict()
    candidates = pd.read_csv(args.candidates, dtype=str)
    candidates["candidate_rank"] = candidates["candidate_rank"].astype(int)
    top1 = candidates[candidates["candidate_rank"] == 1].set_index("analysis_trace_id")
    evidence = defaultdict(lambda: {"first_event": None, "first_error": None, "entry_service": "", "explicit": []})
    scanned = 0
    with args.events.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            scanned += 1
            trace_id = row["analysis_trace_id"]
            if trace_id not in alert_map:
                continue
            item = evidence[trace_id]
            message = (row.get("message") or "").strip()
            event = {"ordinal": int(row.get("event_ordinal") or scanned), "service": (row.get("tc_service") or "").strip(), "level": (row.get("level") or "").strip().upper(), "message": message}
            if item["first_event"] is None:
                item["first_event"] = event
                item["entry_service"] = event["service"]
            if event["level"] == "ERROR" and item["first_error"] is None:
                item["first_error"] = event
            for pattern in EXPLICIT_PATTERNS:
                match = pattern.search(message)
                if match:
                    item["explicit"].append({"service": match.group(1), "ordinal": event["ordinal"], "message": message})
            if scanned % 1_000_000 == 0:
                print(f"[causa] eventos lidos: {scanned:,}", flush=True)

    fields = ["analysis_trace_id", "alert_level", "anomaly_score", "manual_label", "hypothesis_service", "hypothesis_source", "candidate_service_top1", "first_error_service", "entry_service", "explicit_logical_component", "evidence_event_ordinal", "evidence_level", "evidence_message", "limitation"]
    rows = []
    source_counts = defaultdict(int)
    for trace_id in alert_map:
        item = evidence.get(trace_id, {})
        first_error = item.get("first_error")
        explicit = item.get("explicit") or []
        candidate = top1.loc[trace_id, "tc_service"] if trace_id in top1.index else ""
        if explicit:
            hypothesis, source = explicit[0]["service"], "explicit_logical_component"
            evidence_event = explicit[0]
        elif first_error:
            hypothesis, source = first_error["service"], "first_error_service"
            evidence_event = first_error
        elif candidate:
            hypothesis, source = candidate, "candidate_service_top1"
            evidence_event = item.get("first_event") or {"ordinal": "", "level": "", "message": ""}
        else:
            hypothesis, source = item.get("entry_service", ""), "entry_service"
            evidence_event = item.get("first_event") or {"ordinal": "", "level": "", "message": ""}
        source_counts[source] += 1
        rows.append({"analysis_trace_id": trace_id, "alert_level": alert_map[trace_id], "anomaly_score": score_map[trace_id], "manual_label": label_map[trace_id], "hypothesis_service": hypothesis, "hypothesis_source": source, "candidate_service_top1": candidate, "first_error_service": first_error["service"] if first_error else "", "entry_service": item.get("entry_service", ""), "explicit_logical_component": explicit[0]["service"] if explicit else "", "evidence_event_ordinal": evidence_event.get("ordinal", ""), "evidence_level": evidence_event.get("level", ""), "evidence_message": evidence_event.get("message", "")[:1000], "limitation": "hipótese baseada em evidência do trace; não prova causalidade"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, args.output)
    manifest = {"stage": 10, "stage_name": "generate_cause_hypotheses", "generated_at": datetime.now(timezone.utc).isoformat(), "inputs": {"alerts": {"path": str(args.alerts), "sha256": sha256_file(args.alerts)}, "candidates": {"path": str(args.candidates), "sha256": sha256_file(args.candidates)}, "events": {"path": str(args.events), "sha256": sha256_file(args.events)}}, "policy": {"priority": ["explicit_logical_component", "first_error_service", "candidate_service_top1", "entry_service"], "claim": "hypothesis_with_traceable_evidence_not_causal_proof"}, "results": {"events_scanned": scanned, "traces_alerted": len(alert_map), "traces_with_evidence": len(evidence), "hypothesis_source_counts": dict(source_counts), "rows": len(rows)}, "output": {"path": str(args.output), "sha256": sha256_file(args.output)}, "checks": {"traces_nonzero": bool(rows), "one_row_per_alerted_trace": len(rows) == len(alert_map), "output_hash_recorded": True}}
    manifest["accepted"] = all(manifest["checks"].values())
    args.manifest.parent.mkdir(parents=True, exist_ok=True); temporary_manifest = Path(str(args.manifest) + ".tmp"); temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
