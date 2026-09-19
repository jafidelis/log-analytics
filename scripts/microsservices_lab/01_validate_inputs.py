from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


LABELS = {"anomalia", "nao_anomalia", "incerto"}


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        return list(reader.fieldnames), list(reader)


def main() -> None:
    configure_csv_field_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_traces_anonymized.csv"))
    parser.add_argument("--labels", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_trace_manual_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/microservices_sample/microsservices_lab_stage1_manifest.json"))
    args = parser.parse_args()

    event_fields, event_rows = read_rows(args.events)
    label_fields, label_rows = read_rows(args.labels)
    required_events = {"tc_trace_id", "level", "message", "stack_trace", "tc_service"}
    required_labels = {"tc_trace_id", "label"}
    if not required_events.issubset(event_fields):
        raise ValueError(f"Colunas de eventos ausentes: {required_events - set(event_fields)}")
    if not required_labels.issubset(label_fields):
        raise ValueError(f"Colunas de rótulos ausentes: {required_labels - set(label_fields)}")

    event_ids = {row["tc_trace_id"].strip() for row in event_rows if row.get("tc_trace_id", "").strip()}
    label_ids = [row["tc_trace_id"].strip() for row in label_rows]
    label_values = [row["label"].strip() for row in label_rows]
    checks = {
        "event_rows_nonzero": len(event_rows) > 0,
        "labels_count_300": len(label_rows) == 1000,
        "label_ids_nonempty": all(label_ids),
        "label_ids_unique": len(label_ids) == len(set(label_ids)),
        "labels_valid": set(label_values).issubset(LABELS),
        "labels_exist_in_events": set(label_ids).issubset(event_ids),
    }
    manifest = {
        "stage": 1,
        "stage_name": "validate_inputs",
        "events": {"path": str(args.events), "sha256": sha256(args.events),
                   "rows": len(event_rows), "distinct_trace_ids": len(event_ids)},
        "labels": {"path": str(args.labels), "sha256": sha256(args.labels),
                   "rows": len(label_rows),
                   "distribution": {label: label_values.count(label) for label in sorted(LABELS)}},
        "checks": checks,
        "accepted": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["accepted"]:
        raise SystemExit("Validação rejeitada.")


if __name__ == "__main__":
    main()
