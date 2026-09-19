from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import sys
from collections import Counter
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


def aggregate_sorted(path: Path):
    current = None
    counts = Counter()
    for line in path.open(encoding="utf-8"):
        trace_id, template_id = line.rstrip("\n").split("\t")
        template_id = int(template_id)
        if current is not None and trace_id != current:
            yield current, counts
            counts = Counter()
        current = trace_id
        counts[template_id] += 1
    if current is not None:
        yield current, counts


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser(description="Constrói representação esparsa por trace.")
    parser.add_argument("--assignments", type=Path, default=Path("data/processed/microservices_final/microservices_events_drain3_assignments.csv.gz"))
    parser.add_argument("--trace-index", type=Path, default=Path("data/processed/microservices_final/microservices_trace_index.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/microservices_final/microservices_trace_representations.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("microsservices_final/manifests/stage06_trace_representations.manifest.json"))
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("Artefatos da representação já existem; remova-os deliberadamente para reexecutar.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="microservices_trace_sort_") as temp_dir:
        temp = Path(temp_dir)
        unsorted = temp / "assignments.tsv"
        sorted_path = temp / "assignments.sorted.tsv"
        event_count = 0
        with gzip.open(args.assignments, "rt", encoding="utf-8", newline="") as stream, unsorted.open("w", encoding="utf-8") as out:
            for row in csv.DictReader(stream):
                out.write(f"{row['analysis_trace_id']}\t{row['template_id']}\n")
                event_count += 1
                if event_count % 1_000_000 == 0:
                    print(f"[representação] preparado para ordenação: {event_count:,} eventos", flush=True)
        subprocess.run(["sort", "-T", str(temp), "-k1,1", "-k2,2n", str(unsorted), "-o", str(sorted_path)], check=True)

        fields = ["analysis_trace_id", "analysis_trace_id_source", "event_count", "info_count", "warn_count", "error_count", "other_count", "error_rate", "warn_rate", "has_stack_trace", "service_count", "unique_template_count", "unk_count", "top_template_id", "top_template_count", "template_counts"]
        traces = 0
        matched_index = 0
        temporary = Path(str(args.output) + ".tmp")
        with temporary.open("wb") as raw_stream, gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed, __import__("io").TextIOWrapper(compressed, encoding="utf-8", newline="") as text_stream:
            writer = csv.DictWriter(text_stream, fieldnames=fields)
            writer.writeheader()
            index_stream = args.trace_index.open(encoding="utf-8", newline="")
            index_reader = csv.DictReader(index_stream)
            index_row = next(index_reader, None)
            for trace_id, counts in aggregate_sorted(sorted_path):
                while index_row is not None and index_row["analysis_trace_id"] < trace_id:
                    index_row = next(index_reader, None)
                if index_row is None or index_row["analysis_trace_id"] != trace_id:
                    raise ValueError(f"Trace sem índice: {trace_id}")
                ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                unk_count = counts.get(0, 0)
                writer.writerow({
                    "analysis_trace_id": trace_id, "analysis_trace_id_source": index_row["analysis_trace_id_source"],
                    "event_count": index_row["event_count"], "info_count": index_row["info_count"], "warn_count": index_row["warn_count"], "error_count": index_row["error_count"], "other_count": index_row["other_count"],
                    "error_rate": int(index_row["error_count"]) / int(index_row["event_count"]), "warn_rate": int(index_row["warn_count"]) / int(index_row["event_count"]), "has_stack_trace": int(int(index_row["stack_trace_count"]) > 0), "service_count": index_row["service_count"],
                    "unique_template_count": len(counts), "unk_count": unk_count, "top_template_id": ordered[0][0], "top_template_count": ordered[0][1], "template_counts": " ".join(f"{key}:{value}" for key, value in ordered),
                })
                traces += 1
                matched_index += 1
                if traces % 100_000 == 0:
                    print(f"[representação] traces materializados: {traces:,}", flush=True)
            index_stream.close()
        os.replace(temporary, args.output)

    manifest = {
        "stage": 6, "stage_name": "build_sparse_trace_representations", "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"assignments": {"path": str(args.assignments), "sha256": sha256_file(args.assignments)}, "trace_index": {"path": str(args.trace_index), "sha256": sha256_file(args.trace_index)}},
        "representation": {"type": "sparse_template_counts_plus_structural_features", "template_counts_format": "template_id:count separated by spaces", "unk_template_id": 0, "dense_template_matrix": False},
        "results": {"assignment_events": event_count, "traces": traces, "traces_joined_to_index": matched_index},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "checks": {"events_nonzero": event_count > 0, "traces_nonzero": traces > 0, "trace_index_join_complete": matched_index == traces, "output_hash_recorded": True},
    }
    manifest["accepted"] = all(manifest["checks"].values())
    temporary_manifest = Path(str(args.manifest) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
