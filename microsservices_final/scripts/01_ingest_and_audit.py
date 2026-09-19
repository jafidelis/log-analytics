from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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


def audit_file(path: Path, expected_schema: list[str] | None) -> tuple[dict, list[str], set[str]]:
    rows = 0
    empty_trace_ids = 0
    empty_tc_with_trace_id = 0
    fallback_trace_ids: set[str] = set()
    trace_ids: set[str] = set()
    schema: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        schema = list(reader.fieldnames or [])
        if not schema:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        if expected_schema is not None and schema != expected_schema:
            raise ValueError(f"Schema divergente em {path}: {schema}")
        if "tc_trace_id" not in schema:
            raise ValueError(f"Campo tc_trace_id ausente: {path}")
        for row in reader:
            rows += 1
            trace_id = (row.get("tc_trace_id") or "").strip()
            if trace_id:
                trace_ids.add(trace_id)
            else:
                empty_trace_ids += 1
                fallback = (row.get("traceId") or "").strip()
                if fallback:
                    empty_tc_with_trace_id += 1
                    fallback_trace_ids.add(fallback)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "distinct_trace_ids": len(trace_ids),
        "rows_without_trace_id": empty_trace_ids,
        "rows_without_tc_trace_id_with_traceId": empty_tc_with_trace_id,
        "distinct_traceId_fallback_values": len(fallback_trace_ids),
    }, schema, trace_ids


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser(description="Audita os logs originais em streaming.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/microservices"))
    parser.add_argument("--output", type=Path, default=Path("microsservices_final/manifests/stage01_ingest_audit.manifest.json"))
    args = parser.parse_args()

    paths = sorted(args.input.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {args.input}")

    files = []
    schema: list[str] | None = None
    trace_ids: set[str] = set()
    for index, path in enumerate(paths, start=1):
        print(f"[auditoria] {index}/{len(paths)} {path}", flush=True)
        details, current_schema, current_trace_ids = audit_file(path, schema)
        schema = current_schema if schema is None else schema
        files.append(details)
        trace_ids.update(current_trace_ids)

    total_rows = sum(item["rows"] for item in files)
    empty_rows = sum(item["rows_without_trace_id"] for item in files)
    fallback_rows = sum(item["rows_without_tc_trace_id_with_traceId"] for item in files)
    manifest = {
        "stage": 1,
        "stage_name": "ingest_and_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(args.input),
        "files": files,
        "schema": schema,
        "results": {
            "files": len(files),
            "rows": total_rows,
            "distinct_trace_ids": len(trace_ids),
            "rows_without_trace_id": empty_rows,
            "rows_without_tc_trace_id_with_traceId": fallback_rows,
            "rows_without_any_trace_identifier": empty_rows - fallback_rows,
        },
        "checks": {
            "input_files_found": bool(files),
            "schema_present": bool(schema),
            "tc_trace_id_present": "tc_trace_id" in (schema or []),
            "rows_nonzero": total_rows > 0,
            "trace_ids_nonempty": empty_rows == 0,
            "fallback_trace_id_profiled": True,
            "file_hashes_recorded": all(len(item["sha256"]) == 64 for item in files),
        },
    }
    manifest["accepted"] = all(manifest["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
