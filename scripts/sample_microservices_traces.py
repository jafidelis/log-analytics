from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b[A-Fa-f0-9]{16,}\b")

SENSITIVE_FIELDS = {
    "message",
    "stack_trace",
    "tc_user",
    "tc_uri_url",
    "tc_trace_call_stack",
}


def configure_csv_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def anonymize_text(
    value: str,
    mappings: dict[str, str],
) -> str:
    def replace(pattern, prefix, text):
        def repl(match):
            original = match.group(0)

            if original not in mappings:
                mappings[original] = (
                    f"<{prefix}_{len(mappings) + 1}>"
                )

            return mappings[original]

        return pattern.sub(repl, text)

    value = replace(JWT_RE, "JWT", value)
    value = replace(EMAIL_RE, "EMAIL", value)
    value = replace(IP_RE, "IP", value)
    value = replace(HEX_RE, "HEX", value)

    # UUIDs permanecem intactos.
    return value


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


def get_trace_id(row: dict[str, str]) -> str:
    return (row.get("tc_trace_id") or "").strip()


def main() -> None:
    configure_csv_limit()

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
            "microservices_traces_anonymized.csv"
        ),
    )
    parser.add_argument(
        "--percent-sample",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="20260913",
    )

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input.rglob("*.csv"))

    if not paths:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {args.input}"
        )

    # Primeiro passo: contar eventos por trace e registrar severidades.
    trace_stats: dict[str, dict[str, int | bool]] = {}
    total_rows = 0
    rows_without_trace = 0

    print("[passo 1] analisando traces")

    for path, row in iter_rows(paths):
        total_rows += 1

        if total_rows % 100_000 == 0:
            print(
                f"[progresso] {total_rows:,} linhas processadas",
                flush=True,
            )

        trace_id = get_trace_id(row)

        if not trace_id:
            rows_without_trace += 1
            continue

        level = (row.get("level") or "").strip().upper()

        if trace_id not in trace_stats:
            trace_stats[trace_id] = {
                "rows": 0,
                "has_error": False,
                "has_warn": False,
                "has_other": False,
            }

        stats = trace_stats[trace_id]
        stats["rows"] = int(stats["rows"]) + 1

        if level == "ERROR":
            stats["has_error"] = True
        elif level == "WARN":
            stats["has_warn"] = True
        elif level != "INFO":
            stats["has_other"] = True

    target_rows = int(total_rows * args.percent_sample)

    # Ordenação determinística dos traces.
    ranked_trace_ids = sorted(
        trace_stats,
        key=lambda trace_id: hashlib.sha256(
            f"{args.seed}\0{trace_id}".encode("utf-8")
        ).hexdigest(),
    )

    # Seleciona traces completos até atingir aproximadamente 1%.
    selected_trace_ids = []
    selected_rows = 0

    for trace_id in ranked_trace_ids:
        trace_rows = int(trace_stats[trace_id]["rows"])

        if selected_rows >= target_rows:
            break

        selected_trace_ids.append(trace_id)
        selected_rows += trace_rows

    selected_set = set(selected_trace_ids)

    print(f"[passo 1] linhas totais: {total_rows:,}")
    print(f"[passo 1] linhas-alvo: {target_rows:,}")
    print(f"[passo 1] linhas selecionadas: {selected_rows:,}")
    print(
        f"[passo 1] traces distintos: "
        f"{len(trace_stats):,}"
    )
    print(
        f"[passo 1] traces selecionados: "
        f"{len(selected_set):,}"
    )

    # Segundo passo: extrair todos os eventos dos traces selecionados.
    print("[passo 2] extraindo traces completos")

    mappings = {}
    output_rows = []

    for path, row in iter_rows(paths):
        trace_id = get_trace_id(row)

        if trace_id not in selected_set:
            continue

        row_copy = dict(row)

        for field in SENSITIVE_FIELDS:
            if field in row_copy:
                row_copy[field] = anonymize_text(
                    row_copy[field] or "",
                    mappings,
                )

        output_rows.append(row_copy)

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fieldnames = list(output_rows[0].keys())
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    selected_stats = [
        trace_stats[trace_id]
        for trace_id in selected_trace_ids
    ]

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "percent_sample": args.percent_sample,
        "total_rows": total_rows,
        "target_rows": target_rows,
        "sample_rows": len(output_rows),
        "distinct_tc_trace_id": len(selected_set),
        "tc_trace_id_with_at_least_one_ERROR": sum(
            bool(item["has_error"])
            for item in selected_stats
        ),
        "tc_trace_id_with_at_least_one_WARN": sum(
            bool(item["has_warn"])
            for item in selected_stats
        ),
        "tc_trace_id_with_only_INFO": sum(
            not item["has_error"]
            and not item["has_warn"]
            and not item["has_other"]
            for item in selected_stats
        ),
        "rows_without_trace": rows_without_trace,
        "uuid_preserved": True,
        "tc_trace_id_preserved": True,
        "anonymized_fields": sorted(SENSITIVE_FIELDS),
    }

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()