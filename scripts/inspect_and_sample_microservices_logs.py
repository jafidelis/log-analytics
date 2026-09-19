from __future__ import annotations

import argparse
import csv
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import sys


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
HEX_RE = re.compile(r"\b[A-Fa-f0-9]{16,}\b")

def configure_csv_field_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10


def anonymize_text(value: str, mappings: dict[str, str]) -> str:
    """Substitui dados sensíveis mantendo consistência na amostra."""

    def replace(pattern, prefix, text):
        def repl(match):
            original = match.group(0)
            if original not in mappings:
                mappings[original] = f"<{prefix}_{len(mappings) + 1}>"
            return mappings[original]

        return pattern.sub(repl, text)

    value = replace(JWT_RE, "JWT", value)
    value = replace(EMAIL_RE, "EMAIL", value)
    # value = replace(UUID_RE, "UUID", value)
    value = replace(IP_RE, "IP", value)
    value = replace(HEX_RE, "HEX", value)

    return value


def csv_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.csv"))

    if files:
        return files

    return []


def iter_rows_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv"):
                continue

            with archive.open(name) as stream:
                text_stream = (line.decode("utf-8", errors="replace")
                                for line in stream)
                reader = csv.DictReader(text_stream)

                for row in reader:
                    yield name, row


def main() -> None:
    configure_csv_field_limit()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/microservices"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/microservices_sample/"
            "microservices_sample_anonymized.csv"
        ),
    )
    parser.add_argument("--sample-per-level", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260912)

    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    files = csv_files(args.input)
    sources = []

    if files:
        sources = [("csv", path) for path in files]
    else:
        sources = [
            ("zip", path)
            for path in sorted(args.input.glob("*.zip"))
        ]

    if not sources:
        raise FileNotFoundError(
            f"Nenhum CSV ou ZIP encontrado em {args.input}"
        )

    counters = Counter()
    reservoirs = defaultdict(list)
    mappings = {}

    def process_row(source_name: str, row: dict[str, str]):
        level = (row.get("level") or "UNKNOWN").strip().upper()
        counters["total_rows"] += 1
        if counters["total_rows"] % 100_000 == 0:
            print(
                f"[progresso] {counters['total_rows']:,} linhas processadas",
                flush=True,
            )
        counters[f"level_{level}"] += 1

        counters[level] += 1
        bucket = reservoirs[level]

        item = {
            "@timestamp": row.get("@timestamp", ""),
            "level": level,
            "level_value": row.get("level_value", ""),
            "tc_service": row.get("tc_service", ""),
            "tc_app_name": row.get("tc_app_name", ""),
            "tc_domain": row.get("tc_domain", ""),
            "tc_instance": anonymize_text(
                row.get("tc_instance", ""), mappings
            ),
            "traceId": anonymize_text(
                row.get("traceId", ""), mappings
            ),
            "tc_trace_id": anonymize_text(
                row.get("tc_trace_id", ""), mappings
            ),
            "tc_uri_method": row.get("tc_uri_method", ""),
            "tc_uri_path": row.get("tc_uri_path", ""),
            "tc_version": row.get("tc_version", ""),
            "logger_name": row.get("logger_name", ""),
            "message": anonymize_text(
                row.get("message", ""), mappings
            ),
            "stack_trace": anonymize_text(
                row.get("stack_trace", ""), mappings
            ),
            "source_file": source_name,
        }

        if len(bucket) < args.sample_per_level:
            bucket.append(item)
            return

        position = rng.randrange(counters[level])
        if position < args.sample_per_level:
            bucket[position] = item

    for source_type, source_path in sources:
        print(f"[início] processando: {source_path}", flush=True)
        if source_type == "csv":
            with source_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                reader = csv.DictReader(stream)

                for row in reader:
                    process_row(str(source_path), row)
        else:
            for source_name, row in iter_rows_from_zip(source_path):
                process_row(f"{source_path}:{source_name}", row)

    sample = [
        item
        for level in sorted(reservoirs)
        for item in reservoirs[level]
    ]

    sample.sort(key=lambda item: item["@timestamp"])

    fieldnames = list(sample[0].keys())

    with args.output.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample)

    manifest_path = args.output.with_suffix(".manifest.json")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "sample_per_level": args.sample_per_level,
        "sources": [str(path) for _, path in sources],
        "rows": dict(counters),
        "sample_rows": len(sample),
        "sample_levels": {
            level: len(rows)
            for level, rows in sorted(reservoirs.items())
        },
        "anonymization": {
            "email": True,
            # "uuid": True,
            "ip": True,
            "jwt": True,
            "long_hex_tokens": True,
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()