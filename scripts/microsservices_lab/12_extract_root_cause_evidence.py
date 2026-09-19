"""Etapa 12 — evidências do log bruto para validação manual da causa raiz.

Para cada trace da ficha da Etapa 11, varre ``microservices_traces_anonymized.csv``
por ``tc_trace_id`` e extrai:

- tenants observados (``tc_tenant``) e o tenant do primeiro evento ERROR;
- usuário disparador, serviço/primitivo de entrada e janela temporal;
- contagem de ERROR/WARN, serviços com ERROR, primeiro ERROR e exceção;
- padrões de WARN mais frequentes (mensagem normalizada).

Gera dois arquivos:

- ``stage12_root_cause_evidence.csv`` — somente colunas automáticas (sobrescrito);
- ``stage12_root_cause_manual_review.csv`` — colunas automáticas + colunas manuais.
  Se o arquivo já existir, as colunas manuais são preservadas por ``tc_trace_id``
  e apenas as automáticas são atualizadas.

Colunas manuais (preenchidas pelo revisor):

- ``tc_tenant_manual``: tenant validado da causa raiz (a partir do log bruto);
- ``tenant_validation``: confirmado_unico | confirmado_multi_tenant | divergente | nao_aplicavel;
- ``root_service_manual``: serviço raiz (ou ``nenhum``);
- ``root_cause_category_manual``: categoria da causa (vocabulário livre, ver README);
- ``evidence_manual``: evidência textual (timestamp, logger, mensagem);
- ``manual_label_review``: anomalia | nao_anomalia | incerto;
- ``reviewer_notes``, ``reviewer``, ``reviewed_at``, ``review_status`` (proposto | confirmado).

As colunas ``root_in_top3`` e ``root_candidate_rank`` são recalculadas a cada execução
comparando ``root_service_manual`` com ``top_candidate_1..3``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MANUAL_FIELDS = [
    "tc_tenant_manual",
    "tenant_validation",
    "root_service_manual",
    "root_cause_category_manual",
    "evidence_manual",
    "manual_label_review",
    "reviewer_notes",
    "reviewer",
    "reviewed_at",
    "review_status",
]

SHEET_FIELDS = [
    "tc_trace_id", "anomaly_score", "alert_level", "event_count", "service_count",
    "top_candidate_1", "top_candidate_2", "top_candidate_3",
]

AUTO_FIELDS = [
    "tenants_observed", "tenant_count", "trigger_user", "entry_service", "entry_primitive",
    "window_start", "window_end", "duration_ms", "raw_event_count",
    "error_count", "warn_count", "error_services",
    "first_error_timestamp", "first_error_service", "first_error_tenant", "first_error_logger",
    "first_error_primitive", "first_error_message", "first_exception",
    "top_warn_patterns", "previous_manual_label",
]

DERIVED_FIELDS = ["root_in_top3", "root_candidate_rank"]

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
QUOTED_RE = re.compile(r"'[^']*'")
NUMBER_RE = re.compile(r"\d+")


def configure_csv_limit() -> None:
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
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ts(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    head, _, tail = value.partition("+")
    if "." in head:
        base, frac = head.split(".", 1)
        head = f"{base}.{frac[:6].ljust(6, '0')}"
    try:
        return datetime.fromisoformat(f"{head}+{tail}" if tail else head)
    except ValueError:
        return None


def normalize_message(message: str) -> str:
    text = UUID_RE.sub("<UUID>", message)
    text = QUOTED_RE.sub("'…'", text)
    text = NUMBER_RE.sub("N", text)
    return " ".join(text.split())[:80]


def short_logger(logger: str) -> str:
    return logger.rsplit(".", 1)[-1] if logger else ""


def exception_head(stack_trace: str) -> str:
    first = stack_trace.strip().splitlines()[0] if stack_trace.strip() else ""
    return " ".join(first.split())[:200]


def build_evidence(rows: list[dict[str, str]], previous_label: str) -> dict[str, str]:
    rows = sorted(rows, key=lambda row: row["@timestamp"])
    tenants = Counter(row["tc_tenant"] for row in rows)
    users = Counter(row["tc_user"] for row in rows)
    errors = [row for row in rows if row["level"] == "ERROR"]
    warns = [row for row in rows if row["level"] == "WARN"]
    first = rows[0]
    start, end = parse_ts(first["@timestamp"]), parse_ts(rows[-1]["@timestamp"])
    duration = "" if not (start and end) else str(int((end - start).total_seconds() * 1000))
    warn_patterns = Counter(
        f"{row['tc_service']}/{short_logger(row['logger_name'])}: {normalize_message(row['message'])}"
        for row in warns
    )
    first_error = errors[0] if errors else None
    return {
        "tenants_observed": "|".join(f"{tenant}:{count}" for tenant, count in tenants.most_common()),
        "tenant_count": str(len(tenants)),
        "trigger_user": "|".join(user for user, _ in users.most_common(2)),
        "entry_service": first["tc_service"],
        "entry_primitive": first["tc_primitive"],
        "window_start": first["@timestamp"],
        "window_end": rows[-1]["@timestamp"],
        "duration_ms": duration,
        "raw_event_count": str(len(rows)),
        "error_count": str(len(errors)),
        "warn_count": str(len(warns)),
        "error_services": "|".join(
            f"{service}:{count}"
            for service, count in Counter(row["tc_service"] for row in errors).most_common()
        ),
        "first_error_timestamp": first_error["@timestamp"] if first_error else "",
        "first_error_service": first_error["tc_service"] if first_error else "",
        "first_error_tenant": first_error["tc_tenant"] if first_error else "",
        "first_error_logger": short_logger(first_error["logger_name"]) if first_error else "",
        "first_error_primitive": first_error["tc_primitive"] if first_error else "",
        "first_error_message": " ".join(first_error["message"].split())[:300] if first_error else "",
        "first_exception": exception_head(first_error["stack_trace"]) if first_error else "",
        "top_warn_patterns": " || ".join(
            f"x{count} {pattern}" for pattern, count in warn_patterns.most_common(4)
        ),
        "previous_manual_label": previous_label,
    }


def derive_agreement(row: dict[str, str]) -> dict[str, str]:
    root = (row.get("root_service_manual") or "").strip()
    if not root or root == "nenhum":
        return {"root_in_top3": "", "root_candidate_rank": ""}
    candidates = [row.get(f"top_candidate_{index}", "") for index in (1, 2, 3)]
    for index, candidate in enumerate(candidates, start=1):
        if candidate == root:
            return {"root_in_top3": "sim", "root_candidate_rank": str(index)}
    return {"root_in_top3": "nao", "root_candidate_rank": ""}


def main() -> None:
    configure_csv_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, default=Path(
        "artifacts/microsservices_lab/stage11_root_cause_annotation.csv"))
    parser.add_argument("--events", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_traces_anonymized.csv"))
    parser.add_argument("--labels", type=Path, default=Path(
        "data/processed/microservices_sample/microservices_trace_manual_labels.csv"))
    parser.add_argument("--evidence-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_evidence.csv"))
    parser.add_argument("--review-output", type=Path, default=Path(
        "artifacts/microsservices_lab/stage12_root_cause_manual_review.csv"))
    args = parser.parse_args()

    with args.annotation.open("r", encoding="utf-8-sig", newline="") as stream:
        sheet = list(csv.DictReader(stream))
    trace_ids = [row["tc_trace_id"] for row in sheet]
    wanted = set(trace_ids)

    previous_labels: dict[str, str] = {}
    if args.labels.exists():
        with args.labels.open("r", encoding="utf-8-sig", newline="") as stream:
            previous_labels = {row["tc_trace_id"]: row["label"] for row in csv.DictReader(stream)}

    events: dict[str, list[dict[str, str]]] = defaultdict(list)
    scanned = 0
    with args.events.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            scanned += 1
            if row["tc_trace_id"] in wanted:
                events[row["tc_trace_id"]].append(row)

    existing_manual: dict[str, dict[str, str]] = {}
    if args.review_output.exists():
        with args.review_output.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                existing_manual[row["tc_trace_id"]] = {field: row.get(field, "") for field in MANUAL_FIELDS}

    evidence_rows, review_rows = [], []
    missing = []
    for sheet_row in sheet:
        trace_id = sheet_row["tc_trace_id"]
        base = {field: sheet_row.get(field, "") for field in SHEET_FIELDS}
        if not events[trace_id]:
            missing.append(trace_id)
            auto = {field: "" for field in AUTO_FIELDS}
        else:
            auto = build_evidence(events[trace_id], previous_labels.get(trace_id, ""))
        evidence_rows.append({**base, **auto})
        manual = existing_manual.get(trace_id, {field: "" for field in MANUAL_FIELDS})
        review = {**base, **auto, **manual}
        review.update(derive_agreement(review))
        review_rows.append(review)

    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SHEET_FIELDS + AUTO_FIELDS)
        writer.writeheader()
        writer.writerows(evidence_rows)

    review_fields = SHEET_FIELDS + AUTO_FIELDS + MANUAL_FIELDS + DERIVED_FIELDS
    with args.review_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_rows)

    filled = [row for row in review_rows if (row.get("root_service_manual") or "").strip()]
    with_root = [row for row in filled if row["root_service_manual"] != "nenhum"]
    report = {
        "stage": 12,
        "stage_name": "extract_root_cause_evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "annotation": str(args.annotation),
            "events": str(args.events),
            "events_sha256": sha256(args.events),
            "labels": str(args.labels) if args.labels.exists() else None,
        },
        "events_scanned": scanned,
        "traces_requested": len(trace_ids),
        "traces_found": len(trace_ids) - len(missing),
        "traces_missing": missing,
        "raw_events_matched": sum(len(rows) for rows in events.values()),
        "multi_tenant_traces": [
            row["tc_trace_id"] for row in evidence_rows if row["tenant_count"] not in ("", "1")
        ],
        "traces_with_error": sum(1 for row in evidence_rows if row["error_count"] not in ("", "0")),
        "manual_review": {
            "filled": len(filled),
            "with_root_service": len(with_root),
            "root_in_top3": sum(1 for row in with_root if row["root_in_top3"] == "sim"),
            "root_rank_distribution": dict(Counter(row["root_candidate_rank"] for row in with_root)),
            "labels": dict(Counter(row["manual_label_review"] for row in filled)),
            "categories": dict(Counter(row["root_cause_category_manual"] for row in filled)),
            "tenant_validation": dict(Counter(row["tenant_validation"] for row in filled)),
            "review_status": dict(Counter(row["review_status"] for row in filled)),
        },
        "outputs": {
            "evidence": str(args.evidence_output),
            "manual_review": str(args.review_output),
        },
        "manual_fields": MANUAL_FIELDS,
    }
    manifest = args.review_output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
