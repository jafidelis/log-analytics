from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


TERMS = {
    "has_exception_term": re.compile(r"\bexception\b", re.I),
    "has_failed_term": re.compile(r"\bfailed\b", re.I),
    "has_timeout_term": re.compile(r"\btimeout\b", re.I),
    "has_error_term": re.compile(r"\berror\b", re.I),
}


def configure_csv_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def iter_rows(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        for row in reader:
            yield row


def main() -> None:
    configure_csv_limit()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/microservices_sample/"
            "microservices_traces_anonymized.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/microservices_sample/"
            "microservices_trace_signals.csv"
        ),
    )

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    traces: dict[str, dict] = {}
    total_rows = 0

    print("[início] extraindo sinais por trace")

    for row in iter_rows(args.input):
        total_rows += 1

        if total_rows % 10_000 == 0:
            print(
                f"[progresso] {total_rows:,} eventos processados",
                flush=True,
            )

        trace_id = (row.get("tc_trace_id") or "").strip()

        if not trace_id:
            continue

        if trace_id not in traces:
            traces[trace_id] = {
                "event_count": 0,
                "info_count": 0,
                "warn_count": 0,
                "error_count": 0,
                "has_stack_trace": 0,
                "services": set(),
                **{name: 0 for name in TERMS},
            }

        trace = traces[trace_id]

        level = (row.get("level") or "").strip().upper()
        message = row.get("message") or ""
        stack_trace = row.get("stack_trace") or ""
        text = f"{message} {stack_trace}"

        trace["event_count"] += 1

        if level == "INFO":
            trace["info_count"] += 1
        elif level == "WARN":
            trace["warn_count"] += 1
        elif level == "ERROR":
            trace["error_count"] += 1

        if stack_trace.strip():
            trace["has_stack_trace"] = 1

        service = (row.get("tc_service") or "").strip()

        if service:
            trace["services"].add(service)

        for name, pattern in TERMS.items():
            if pattern.search(text):
                trace[name] = 1

    fieldnames = [
        "tc_trace_id",
        "event_count",
        "info_count",
        "warn_count",
        "error_count",
        "has_stack_trace",
        "service_count",
        *TERMS.keys(),
    ]

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for trace_id in sorted(traces):
            trace = traces[trace_id]

            writer.writerow(
                {
                    "tc_trace_id": trace_id,
                    "event_count": trace["event_count"],
                    "info_count": trace["info_count"],
                    "warn_count": trace["warn_count"],
                    "error_count": trace["error_count"],
                    "has_stack_trace": trace["has_stack_trace"],
                    "service_count": len(trace["services"]),
                    **{
                        name: trace[name]
                        for name in TERMS
                    },
                }
            )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "events_processed": total_rows,
        "traces_with_id": len(traces),
        "signals": {
            "traces_with_error": sum(
                trace["error_count"] > 0
                for trace in traces.values()
            ),
            "traces_with_warn": sum(
                trace["warn_count"] > 0
                for trace in traces.values()
            ),
            "traces_with_stack_trace": sum(
                trace["has_stack_trace"] == 1
                for trace in traces.values()
            ),
            **{
                name: sum(
                    trace[name] == 1
                    for trace in traces.values()
                )
                for name in TERMS
            },
        },
    }

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()