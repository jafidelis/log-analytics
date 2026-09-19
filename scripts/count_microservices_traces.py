from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def configure_csv_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def iter_rows(paths: list[Path]):
    for path in paths:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)

            for row in reader:
                yield path, row


def main() -> None:
    configure_csv_limit()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/microservices"),
    )
    args = parser.parse_args()

    paths = sorted(args.input.rglob("*.csv"))

    if not paths:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {args.input}"
        )

    # Bit 1: possui ERROR
    # Bit 2: possui WARN
    # Bit 4: possui outro nível diferente de INFO
    trace_flags: dict[str, int] = {}

    total_rows = 0
    rows_without_trace = 0
    level_counts = Counter()

    for path, row in iter_rows(paths):
        total_rows += 1

        if total_rows % 100_000 == 0:
            print(
                f"[progresso] {total_rows:,} linhas processadas",
                flush=True,
            )

        trace_id = (row.get("tc_trace_id") or "").strip()
        level = (row.get("level") or "").strip().upper()

        level_counts[level] += 1

        if not trace_id:
            rows_without_trace += 1
            continue

        flags = trace_flags.get(trace_id, 0)

        if level == "ERROR":
            flags |= 1
        elif level == "WARN":
            flags |= 2
        elif level != "INFO":
            flags |= 4

        trace_flags[trace_id] = flags

    result = {
        "total_rows": total_rows,
        "distinct_tc_trace_id": len(trace_flags),
        "tc_trace_id_with_at_least_one_ERROR": sum(
            bool(flags & 1)
            for flags in trace_flags.values()
        ),
        "tc_trace_id_with_at_least_one_WARN": sum(
            bool(flags & 2)
            for flags in trace_flags.values()
        ),
        "tc_trace_id_with_only_INFO": sum(
            flags == 0
            for flags in trace_flags.values()
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()