from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
URL_RE = re.compile(r"https?://[^\s,)] +".replace(" ", ""))
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b[A-Fa-f0-9]{16,}\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")
WS_RE = re.compile(r"\s+")


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


def prepare_message(value: str) -> str:
    value = WS_RE.sub(" ", value.strip())
    for pattern, token in ((UUID_RE, "<UUID>"), (TIMESTAMP_RE, "<TIMESTAMP>"), (URL_RE, "<URL>"), (EMAIL_RE, "<EMAIL>"), (IP_RE, "<IP>"), (HEX_RE, "<HEX>")):
        value = pattern.sub(token, value)
    return NUMBER_RE.sub("<NUM>", value)


def build_parser(state: Path) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.drain_sim_th = 0.6
    config.drain_extra_delimiters = []
    config.masking_instructions = []
    config.parametrize_numeric_tokens = False
    return TemplateMiner(persistence_handler=FilePersistence(str(state)), config=config)


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser(description="Transforma eventos usando Drain3 congelado.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/microservices_final/microservices_events_anonymized_traceable.csv"))
    parser.add_argument("--state", type=Path, default=Path("data/processed/microservices_final/microservices_drain3_state.bin"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/microservices_final/microservices_events_drain3_assignments.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage05_drain3_transformation.manifest.json"))
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Artefatos da transformação já existem; remova-os deliberadamente para reexecutar.")
    model = build_parser(args.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    header = ["event_ordinal", "analysis_trace_id", "analysis_trace_id_source", "@timestamp", "level", "tc_service", "template_id", "template"]
    counts = Counter()
    events = 0
    started = perf_counter()
    temporary = Path(str(args.output) + ".tmp")
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_stream:
                writer = csv.DictWriter(text_stream, fieldnames=header)
                writer.writeheader()
                with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        events += 1
                        message = prepare_message(row.get("message") or "")
                        cluster = model.match(message, full_search_strategy="always") if message else None
                        template_id = cluster.cluster_id if cluster is not None else 0
                        template = cluster.get_template() if cluster is not None else "UNK_TEMPLATE"
                        counts["matched" if cluster is not None else "unknown"] += 1
                        writer.writerow({"event_ordinal": events, "analysis_trace_id": row["analysis_trace_id"], "analysis_trace_id_source": row["analysis_trace_id_source"], "@timestamp": row.get("@timestamp", ""), "level": row.get("level", ""), "tc_service": row.get("tc_service", ""), "template_id": template_id, "template": template})
                        if events % 500_000 == 0:
                            elapsed = perf_counter() - started
                            print(f"[drain3] transformação {events:,} eventos | matched={counts['matched']:,} | unk={counts['unknown']:,} | taxa={events / elapsed:,.0f}/s | decorrido={elapsed / 60:.1f} min", flush=True)
    os.replace(temporary, args.output)
    manifest = {"stage": 5, "stage_name": "transform_with_frozen_drain3", "generated_at": datetime.now(timezone.utc).isoformat(), "inputs": {"events": {"path": str(args.input), "sha256": sha256_file(args.input)}, "state": {"path": str(args.state), "sha256": sha256_file(args.state)}}, "results": {"events": events, "matched": counts["matched"], "unknown": counts["unknown"], "unknown_rate": counts["unknown"] / events if events else 0.0, "state_changed_during_match": False}, "output": {"path": str(args.output), "sha256": sha256_file(args.output)}, "checks": {"events_nonzero": events > 0, "event_counts_conserved": counts["matched"] + counts["unknown"] == events, "state_exists": args.state.exists(), "output_hash_recorded": True}}
    manifest["accepted"] = all(manifest["checks"].values())
    temporary_manifest = Path(str(args.manifest) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
