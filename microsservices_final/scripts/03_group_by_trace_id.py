from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Constrói índice lógico de traces.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_traceable.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/microservices_final/microservices_trace_index.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage03_trace_grouping.manifest.json"))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    stats: dict[str, dict] = {}
    rows = 0
    source_counts = defaultdict(int)
    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"analysis_trace_id", "analysis_trace_id_source", "@timestamp", "level", "tc_service", "stack_trace"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Campos ausentes: {sorted(missing)}")
        for row in reader:
            rows += 1
            if rows % 500_000 == 0:
                print(f"[agrupamento] {rows:,} eventos", flush=True)
            trace_id = (row.get("analysis_trace_id") or "").strip()
            if not trace_id:
                raise ValueError(f"Evento sem analysis_trace_id na linha {rows + 1}")
            source = (row.get("analysis_trace_id_source") or "").strip()
            source_counts[source] += 1
            item = stats.get(trace_id)
            timestamp = (row.get("@timestamp") or "").strip()
            level = (row.get("level") or "").strip().upper()
            service = (row.get("tc_service") or "").strip()
            if item is None:
                item = stats[trace_id] = {
                    "event_count": 0, "first_timestamp": timestamp, "last_timestamp": timestamp,
                    "services": set(), "info_count": 0, "warn_count": 0, "error_count": 0,
                    "other_count": 0, "stack_trace_count": 0, "first_event_ordinal": rows,
                    "last_event_ordinal": rows, "sources": set(),
                }
            item["event_count"] += 1
            item["last_event_ordinal"] = rows
            if timestamp and (not item["first_timestamp"] or timestamp < item["first_timestamp"]):
                item["first_timestamp"] = timestamp
            if timestamp > item["last_timestamp"]:
                item["last_timestamp"] = timestamp
            if service:
                item["services"].add(service)
            item["sources"].add(source)
            if level == "INFO":
                item["info_count"] += 1
            elif level == "WARN":
                item["warn_count"] += 1
            elif level == "ERROR":
                item["error_count"] += 1
            else:
                item["other_count"] += 1
            if (row.get("stack_trace") or "").strip():
                item["stack_trace_count"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["analysis_trace_id", "analysis_trace_id_source", "event_count", "first_timestamp", "last_timestamp", "service_count", "services", "info_count", "warn_count", "error_count", "other_count", "stack_trace_count", "first_event_ordinal", "last_event_ordinal"]
    temporary = Path(str(args.output) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trace_id in sorted(stats):
            item = stats[trace_id]
            writer.writerow({
                "analysis_trace_id": trace_id,
                "analysis_trace_id_source": "mixed" if len(item["sources"]) > 1 else next(iter(item["sources"])),
                "event_count": item["event_count"], "first_timestamp": item["first_timestamp"], "last_timestamp": item["last_timestamp"],
                "service_count": len(item["services"]), "services": "|".join(sorted(item["services"])),
                "info_count": item["info_count"], "warn_count": item["warn_count"], "error_count": item["error_count"],
                "other_count": item["other_count"], "stack_trace_count": item["stack_trace_count"],
                "first_event_ordinal": item["first_event_ordinal"], "last_event_ordinal": item["last_event_ordinal"],
            })
    os.replace(temporary, args.output)
    manifest = {
        "stage": 3, "stage_name": "group_by_analysis_trace_id", "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "results": {"events": rows, "traces": len(stats), "source_counts": dict(source_counts), "event_count_conserved": sum(item["event_count"] for item in stats.values()) == rows},
        "outputs": {"trace_index": {"path": str(args.output), "sha256": sha256_file(args.output), "traces": len(stats)}},
        "checks": {"input_exists": True, "trace_ids_nonempty": bool(stats), "event_count_conserved": sum(item["event_count"] for item in stats.values()) == rows, "index_hash_recorded": True},
    }
    manifest["accepted"] = all(manifest["checks"].values())
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = Path(str(args.manifest) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
