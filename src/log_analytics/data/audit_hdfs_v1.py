from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

LOG_HEADER_PATTERN = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s?"
    r"(?P<content>.*)$"
)
BLOCK_ID_PATTERN = re.compile(r"\bblk_-?\d+\b")
EXPECTED_ZIP_MD5 = "76a24b4d9a6164d543fb275f89773260"
SAMPLE_LIMIT = 10
LABEL_NORMAL = "Normal"
LABEL_ANOMALY = "Anomaly"

@dataclass(frozen=True)
class LogEvent:
    """Cabeçalho e conteúdo de uma linha válida do HDFS.log."""

    line_id: int
    date: str
    time: str
    pid: int
    level: str
    component: str
    content: str

    @property
    def timestamp(self) -> str:
        """Combina data e hora sem descartar os campos originais."""

        value = datetime.strptime(f"{self.date}{self.time}", "%y%m%d%H%M%S")
        return value.isoformat()

@dataclass(frozen=True)
class DatasetExpectations:
    """Valores que caracterizam a versão do dataset sob auditoria."""

    total_log_lines: int
    unique_labels: int
    normal_labels: int
    anomaly_labels: int
    log_sha256: str | None = None
    labels_sha256: str | None = None

HDFS_V1_EXPECTATIONS = DatasetExpectations(
    total_log_lines=11_175_629,
    unique_labels=575_061,
    normal_labels=558_223,
    anomaly_labels=16_838,

log_sha256="0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52",

labels_sha256="1c711ed6c8848fc3243fb4d092f172f31d128c8a6ec7f26ebba72ab931885ed8",
)

def normalize_column_name(value: str) -> str:
    """Remove espaços externos e normaliza a caixa de um cabeçalho."""

    return value.strip().lower()

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula SHA-256 sem carregar o arquivo inteiro em memória."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def load_labels(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Lê e valida o CSV de rótulos, mantendo uma sessão por BlockId."""

    labels: dict[str, str] = {}
    invalid_block_ids = 0
    invalid_labels = 0 
    blank_values = 0
    duplicate_rows = 0
    conflicting_rows = 0
    samples: dict[str, list[dict[str, Any]]] = {
        "invalid_block_ids": [],
        "invalid_labels": [],
        "blank_values": [],
        "duplicates_or_conflicts": [],
    }
    total_rows = 0

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("O CSV de rótulos não possui cabeçalho.")

        normalized = {
            normalize_column_name(name): name for name in reader.fieldnames
        }

        if set(normalized) != {"blockid", "label"}:
            raise ValueError(
                "Esquema inválido no CSV. Esperado: BlockId,Label; "
                f"encontrado: {reader.fieldnames}."
            )

        block_column = normalized["blockid"]
        label_column = normalized["label"]

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            block_id = (row.get(block_column) or "").strip()
            label = (row.get(label_column) or "").strip()

            if not block_id or not label:
                blank_values += 1
                if len(samples["blank_values"]) < SAMPLE_LIMIT:
                    samples["blank_values"].append({"row_number": row_number})
                continue

            if BLOCK_ID_PATTERN.fullmatch(block_id) is None:
                invalid_block_ids += 1
                if len(samples["invalid_block_ids"]) < SAMPLE_LIMIT:
                    samples["invalid_block_ids"].append(
                        {"row_number": row_number, "block_id": block_id}
                    )
                continue

            if label not in {LABEL_NORMAL, LABEL_ANOMALY}:
                invalid_labels += 1
                if len(samples["invalid_labels"]) < SAMPLE_LIMIT:
                    samples["invalid_labels"].append(
                        {"row": row_number, "label": label}
                    )
                continue

            previous = labels.get(block_id)
            if previous is not None:
                duplicate_rows += 1
                if previous != label:
                    conflicting_rows += 1
                if len(samples["duplicates_or_conflicts"]) < SAMPLE_LIMIT:
                    samples["duplicates_or_conflicts"].append(
                        {
                            "row": row_number,
                            "block_id": block_id,
                            "previous_label": previous,
                            "current_label": label,
                        }
                    )
                continue
            labels[block_id] = label

    label_counts = Counter(labels.values())
    metrics = {
        "total_data_rows": total_rows,
        "unique_block_ids": len(labels),
        "normal": label_counts["Normal"],
        "anomaly": label_counts["Anomaly"],
        "blank_values": blank_values,
        "invalid_block_ids": invalid_block_ids,
        "invalid_labels": invalid_labels,
        "duplicate_rows": duplicate_rows,
        "conflicting_rows": conflicting_rows,
        "samples": samples,
    }
    return labels, metrics

def parse_log_line(line: str, line_id: int) -> LogEvent | None:
    """Converte uma linha em LogEvent; retorna None se o cabeçalho não casar."""

    match = LOG_HEADER_PATTERN.fullmatch(line)
    if match is None:
        return None

    values = match.groupdict()
    return LogEvent(
        line_id=line_id,
        date=values["date"],
        time=values["time"],
        pid=int(values["pid"]),
        level=values["level"],
        component=values["component"].strip(),
        content=values["content"],
    )

def extract_block_ids(content: str) -> list[str]:
    """Extrai BlockIds distintos preservando a primeira ordem de aparição."""

    return tuple(dict.fromkeys(BLOCK_ID_PATTERN.findall(content)))

def audit_log(path: Path, labels: dict[str, str]) -> dict[str, Any]:
    """Varre o log em streaming e calcula integridade e associações de sessão."""

    digest = hashlib.sha256()
    observed_block_ids: set[str] = set()
    unlabeled_block_ids: set[str] = set()

    total_lines = 0
    parsed_header_lines = 0
    unparseable_header_lines = 0
    decoding_errors = 0
    lines_with_block_id = 0
    lines_without_block_id = 0
    lines_with_multiple_distinct_ids = 0
    lines_with_repeated_id_mentions = 0
    event_session_associations = 0

    samples: dict[str, list[dict[str, Any]]] = {
        "decoding_errors": [],
        "unparseable_headers": [],
        "lines_without_block_id": [],
    }

    with path.open("rb") as stream:
        for line_id, raw_line in enumerate(stream, start=1):
            total_lines += 1
            digest.update(raw_line)

            try:
                line = raw_line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as error:
                decoding_errors += 1
                if len(samples["decoding_errors"]) < SAMPLE_LIMIT:
                    samples["decoding_errors"].append(
                        {"line_id": line_id, "reason": str(error)}
                    )
                continue

            event = parse_log_line(line, line_id)
            if event is None:
                unparseable_header_lines += 1
                if len(samples["unparseable_headers"]) < SAMPLE_LIMIT:
                    samples["unparseable_headers"].append(
                        {"line_id": line_id, "text": line[:200]}
                    )
                continue

            parsed_header_lines += 1
            all_mentions = BLOCK_ID_PATTERN.findall(event.content)
            block_ids = tuple(dict.fromkeys(all_mentions))

            if not block_ids:
                lines_without_block_id += 1
                if len(samples["lines_without_block_id"]) < SAMPLE_LIMIT:
                    samples["lines_without_block_id"].append(
                        {"line_id": line_id, "text": event.content[:200]}
                    )
                continue

            lines_with_block_id += 1
            if len(block_ids) > 1:
                lines_with_multiple_distinct_ids += 1
            if len(all_mentions) > len(block_ids):
                lines_with_repeated_id_mentions += 1

            event_session_associations += len(block_ids)
            observed_block_ids.update(block_ids)
            unlabeled_block_ids.update(
                block_id for block_id in block_ids if block_id not in labels
            )

    labels_without_event = set(labels).difference(observed_block_ids)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "total_lines": total_lines,
        "parsed_header_lines": parsed_header_lines,
        "unparseable_header_lines": unparseable_header_lines,
        "decoding_errors": decoding_errors,
        "lines_with_block_id": lines_with_block_id,
        "lines_without_block_id": lines_without_block_id,
        "lines_with_multiple_distinct_ids": lines_with_multiple_distinct_ids,
        "lines_with_repeated_id_mentions": lines_with_repeated_id_mentions,
        "event_session_associations": event_session_associations,
        "distinct_block_ids_observed": len(observed_block_ids),
        "block_ids_without_label": len(unlabeled_block_ids),
        "labels_without_event": len(labels_without_event),
        "sample_block_ids_without_label": sorted(unlabeled_block_ids)[:SAMPLE_LIMIT],
        "sample_labels_without_event": sorted(labels_without_event)[:SAMPLE_LIMIT],
        "samples": samples,
    }

def evaluate_acceptance(
    label_metrics: dict[str, Any],
    log_metrics: dict[str, Any],
    expected: DatasetExpectations,
) -> dict[str, Any]:
    """Avalia critérios objetivos e preserva cada resultado no manifesto."""

    checks = {
        "expected_log_line_count": (
            log_metrics["total_lines"] == expected.total_log_lines
        ),
        "all_log_bytes_decode_as_utf8": log_metrics["decoding_errors"] == 0,
        "all_log_headers_parse": log_metrics["unparseable_header_lines"] == 0,
        "log_line_accounting_closes": (
            log_metrics["parsed_header_lines"]
            + log_metrics["unparseable_header_lines"]
            + log_metrics["decoding_errors"]
            == log_metrics["total_lines"]
        ),
        "parsed_line_session_accounting_closes": (
            log_metrics["lines_with_block_id"]
            + log_metrics["lines_without_block_id"]
            == log_metrics["parsed_header_lines"]
        ),
        "expected_unique_labels": (
            label_metrics["unique_block_ids"] == expected.unique_labels
        ),
        "expected_normal_labels": (
            label_metrics["normal"] == expected.normal_labels
        ),
        "expected_anomaly_labels": (
            label_metrics["anomaly"] == expected.anomaly_labels
        ),
        "no_blank_label_values": label_metrics["blank_values"] == 0,
        "no_invalid_label_block_ids": label_metrics["invalid_block_ids"] == 0,
        "no_invalid_label_domain": label_metrics["invalid_labels"] == 0,
        "no_duplicate_label_rows": label_metrics["duplicate_rows"] == 0,
        "no_conflicting_labels": label_metrics["conflicting_rows"] == 0,
        "all_observed_block_ids_have_label": (
            log_metrics["block_ids_without_label"] == 0
        ),
        "all_labels_have_event": log_metrics["labels_without_event"] == 0,
    }
    if expected.log_sha256 is not None:
        checks["expected_log_sha256"] = (
            log_metrics["sha256"] == expected.log_sha256
        )
    if expected.labels_sha256 is not None:
        checks["expected_labels_sha256"] = (
            label_metrics["sha256"] == expected.labels_sha256
        )
    return {"accepted": all(checks.values()), "checks": checks}

def audit_dataset(
    log_path: Path,
    labels_path: Path,
    expected: DatasetExpectations = HDFS_V1_EXPECTATIONS,
) -> dict[str, Any]:
    """Coordena a auditoria e devolve um manifesto serializável em JSON."""

    if not log_path.is_file():
        raise FileNotFoundError(f"Log não encontrado: {log_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"CSV de rótulos não encontrado: {labels_path}")

    labels, label_metrics = load_labels(labels_path)
    log_metrics = audit_log(log_path, labels)
    labels_sha256 = sha256_file(labels_path)
    label_metrics["sha256"] = labels_sha256
    acceptance = evaluate_acceptance(label_metrics, log_metrics, expected)

    return {
        "manifest_version": 1,
        "dataset": {
            "name": "Loghub HDFS_v1",
            "source_archive_md5_expected": EXPECTED_ZIP_MD5,
            "source_archive_md5_previously_verified": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "inputs": {
            "log": {
                "path": log_path.as_posix(),
                "sha256": log_metrics["sha256"],
                "size_bytes": log_metrics["size_bytes"],
            },
            "labels": {
                "path": labels_path.as_posix(),
                "sha256": labels_sha256,
                "size_bytes": labels_path.stat().st_size,
            },
        },
        "parsing_contract": {
            "encoding": "utf-8 strict",
            "block_id_pattern": BLOCK_ID_PATTERN.pattern,
            "duplicate_id_in_line": "one association",
            "multiple_distinct_ids_in_line": "one association per BlockId",
            "unparseable_or_idless_lines": "excluded from sessions and counted",
        },
        "expectations": asdict(expected),
        "labels": label_metrics,
        "log": log_metrics,
        "acceptance": acceptance,
    }

def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Grava o manifesto de forma atômica e cria o diretório de saída."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Define a interface de linha de comando da auditoria."""

    parser = argparse.ArgumentParser(
        description="Audita a integridade do dataset Loghub HDFS_v1."
    )
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Executa a auditoria; código 0 significa aceite, código 1 reprovação."""

    arguments = build_argument_parser().parse_args(argv)
    manifest = audit_dataset(arguments.log, arguments.labels)
    write_manifest(manifest, arguments.output)

    accepted = manifest["acceptance"]["accepted"]
    print(f"Manifesto: {arguments.output}")
    print(f"Aceite: {'APROVADO' if accepted else 'REPROVADO'}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
