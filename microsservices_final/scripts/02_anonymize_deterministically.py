from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b[A-Fa-f0-9]{16,}\b")
SENSITIVE_FIELDS = {"message", "stack_trace", "tc_user", "tc_uri_url", "tc_trace_call_stack"}


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def stable_token(prefix: str, value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\0{prefix}\0{value}".encode()).hexdigest()[:20]
    return f"<{prefix}_{digest}>"


def anonymize_text(value: str, salt: str) -> str:
    for pattern, prefix in ((JWT_RE, "JWT"), (EMAIL_RE, "EMAIL"), (IP_RE, "IP"), (HEX_RE, "HEX")):
        value = pattern.sub(lambda match: stable_token(prefix, match.group(0), salt), value)
    return value


def atomic_writer(path: Path, fieldnames: list[str]):
    temporary = Path(str(path) + ".tmp")
    stream = temporary.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return temporary, stream, writer


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser(description="Anonimiza os logs completos preservando a política de traces.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/microservices"))
    parser.add_argument("--output-traceable", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_traceable.csv"))
    parser.add_argument("--output-untraceable", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_untraceable.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage02_anonymization.manifest.json"))
    parser.add_argument("--salt", default="microservices_final_v1")
    args = parser.parse_args()

    paths = sorted(args.input.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {args.input}")
    args.output_traceable.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    traceable_tmp = untraceable_tmp = None
    traceable_stream = untraceable_stream = None
    traceable_writer = untraceable_writer = None
    schema: list[str] | None = None
    traceable_rows = untraceable_rows = fallback_rows = 0
    source_counts = {"tc_trace_id": 0, "traceId": 0, "none": 0}

    try:
        for file_index, path in enumerate(paths, start=1):
            print(f"[anonimização] {file_index}/{len(paths)} {path}", flush=True)
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                current_schema = list(reader.fieldnames or [])
                if not current_schema or "tc_trace_id" not in current_schema or "traceId" not in current_schema:
                    raise ValueError(f"Schema sem tc_trace_id/traceId: {path}")
                if schema is None:
                    schema = current_schema + ["analysis_trace_id", "analysis_trace_id_source"]
                    traceable_tmp, traceable_stream, traceable_writer = atomic_writer(args.output_traceable, schema)
                    untraceable_tmp, untraceable_stream, untraceable_writer = atomic_writer(args.output_untraceable, schema)
                elif current_schema + ["analysis_trace_id", "analysis_trace_id_source"] != schema:
                    raise ValueError(f"Schema divergente em {path}")

                for row in reader:
                    tc_trace = (row.get("tc_trace_id") or "").strip()
                    fallback = (row.get("traceId") or "").strip()
                    if tc_trace:
                        analysis_id, source = tc_trace, "tc_trace_id"
                    elif fallback:
                        analysis_id, source = fallback, "traceId_fallback"
                        fallback_rows += 1
                    else:
                        analysis_id, source = "", "none"
                    source_counts["tc_trace_id" if source == "tc_trace_id" else "traceId" if source == "traceId_fallback" else "none"] += 1
                    output_row = dict(row)
                    for field in SENSITIVE_FIELDS:
                        if field in output_row:
                            output_row[field] = anonymize_text(output_row[field] or "", args.salt)
                    output_row["analysis_trace_id"] = analysis_id
                    output_row["analysis_trace_id_source"] = source
                    if analysis_id:
                        traceable_writer.writerow(output_row)
                        traceable_rows += 1
                    else:
                        untraceable_writer.writerow(output_row)
                        untraceable_rows += 1
    finally:
        for stream in (traceable_stream, untraceable_stream):
            if stream is not None:
                stream.close()
        if traceable_tmp is not None:
            os.replace(traceable_tmp, args.output_traceable)
        if untraceable_tmp is not None:
            os.replace(untraceable_tmp, args.output_untraceable)

    manifest = {
        "stage": 2,
        "stage_name": "anonymize_deterministically",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(args.input),
        "policy": {
            "primary_trace_key": "tc_trace_id",
            "fallback_trace_key": "traceId",
            "untraceable_events": "preserved_separately_and_excluded_from_trace_analysis",
            "anonymization": "deterministic_sha256_tokens_for_sensitive_text",
        },
        "results": {"traceable_rows": traceable_rows, "untraceable_rows": untraceable_rows, "fallback_trace_rows": fallback_rows, "source_counts": source_counts},
        "outputs": {"traceable": str(args.output_traceable), "untraceable": str(args.output_untraceable)},
        "accepted": traceable_rows > 0 and untraceable_rows >= 0 and traceable_rows + untraceable_rows > 0,
    }
    temporary_manifest = Path(str(args.manifest) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
