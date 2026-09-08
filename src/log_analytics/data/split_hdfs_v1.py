from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping, Sequence

from log_analytics.data.audit_hdfs_v1 import load_labels, sha256_file


LABEL_NORMAL = "Normal"
LABEL_ANOMALY = "Anomaly"
SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
EXPECTED_LABEL_COUNTS = {
    LABEL_NORMAL: 558_223,
    LABEL_ANOMALY: 16_838,
}
DEFAULT_SALT = "hdfs-v1-master-split-v1"
EXPECTED_LABELS_SHA256 = (
    "1c711ed6c8848fc3243fb4d092f172f31d128c8a6ec7f26ebba72ab931885ed8"
)
EXPECTED_LOG_SHA256 = (
    "0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52"
)

@dataclass(frozen=True)
class SplitConfig:
    """Configuração versionada do split mestre."""

    train_ratio: Decimal = Decimal("0.70")
    validation_ratio: Decimal = Decimal("0.15")
    test_ratio: Decimal = Decimal("0.15")
    salt: str = DEFAULT_SALT
    algorithm: str = "stratified_sha256_rank"
    algorithm_version: int = 1

    @property
    def ratios(self) -> dict[str, Decimal]:
        return {
            SPLIT_TRAIN: self.train_ratio,
            SPLIT_VALIDATION: self.validation_ratio,
            SPLIT_TEST: self.test_ratio,
        }

@dataclass(frozen=True)
class SplitAssignment:
    """Atribuição persistente de uma sessão rotulada."""

    block_id: str
    label: str
    split: str

def validate_ratios(ratios: Mapping[str, Decimal]) -> None:
    """Exige os três splits, valores positivos e soma exatamente igual a um."""

    if tuple(ratios) != SPLIT_ORDER:
        raise ValueError(f"Ordem ou domínio de splits inválido: {tuple(ratios)}")

    if any(value <= 0 for value in ratios.values()):
        raise ValueError(f"Todos os splits devem ser positivos: {ratios}")

    if sum(ratios.values(), start=Decimal(0)) != Decimal(1):
        raise ValueError("A soma das proporções deve ser exatamente 1.")

def allocate_largest_remainder(
        total: int,
        ratios: Mapping[str, Decimal],
) -> dict[str, int]:
    """Converte proporções em contagens inteiras que fecham o total."""

    if total < 0:
        raise ValueError("O total não pode ser negativo.")

    validate_ratios(ratios)

    exact = {name: Decimal(total) * ration for name, ration in ratios.items()}
    allocated = {
        name: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for name, value in exact.items()
    }
    missing = total - sum(allocated.values())
    tie_position = {name: index for index, name in enumerate(SPLIT_ORDER)}
    priority = sorted(
        SPLIT_ORDER,
        key=lambda name: (
            -(exact[name] - Decimal(allocated[name])),  # maior resto primeiro
            tie_position[name],  # desempate pela ordem do SPLIT_ORDER
        ),
    )

    for name in priority[:missing]:
        allocated[name] += 1
    return allocated

def stable_digest(block_id: str, salt: str) -> bytes:
    """Produz a chave SHA-256 portátil usada apenas para ordenar sessões."""

    if not block_id:
         raise ValueError("O block_id não pode ser vazio.")
     
    if not salt:
        raise ValueError("O salt não pode ser vazio.")

    payload = f"{salt}\0{block_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()

def create_stratified_split(
    labels: Mapping[str, str],
    config: SplitConfig,        
) -> list[SplitAssignment]:
    """Atribui cada BlockId a um único split, estratificando por rótulo."""

    validate_ratios(config.ratios)
    unknown_labels = set(labels.values()).difference(EXPECTED_LABEL_COUNTS)

    if unknown_labels:
        raise ValueError(f"Rótulos não reconhecidos: {sorted(unknown_labels)}")

    assignments: list[SplitAssignment] = []
    for label in (LABEL_NORMAL, LABEL_ANOMALY):
        block_ids = [
            block_id
            for block_id, current_label in labels.items()
            if current_label == label
        ]

        ordered = sorted(
            block_ids,
            key= lambda block_id: (
                stable_digest(block_id, config.salt),
                block_id,
            ),
        )

        counts = allocate_largest_remainder(len(ordered), config.ratios)
        train_end = counts[SPLIT_TRAIN]
        validation_end = train_end + counts[SPLIT_VALIDATION]

        for index, block_id in enumerate(ordered):
            if index < train_end:
                split = SPLIT_TRAIN
            elif index < validation_end:
                split = SPLIT_VALIDATION
            else:
                split = SPLIT_TEST

            assignments.append(SplitAssignment(block_id, label, split))

    return sorted(assignments, key=lambda item: item.block_id)

def expected_counts(config: SplitConfig) -> dict[str, int]:
    """Calcula as contagens esperadas por split e rótulo."""

    by_label = {
        label: allocate_largest_remainder(count, config.ratios)
        for label, count in EXPECTED_LABEL_COUNTS.items()
    }

    return {
        split: {
            label: by_label[label][split]
            for label in (LABEL_NORMAL, LABEL_ANOMALY)
        }
        for split in SPLIT_ORDER
    }

def count_assignments(
        assignments: Sequence[SplitAssignment],
) -> dict[str, dict[str, int]]:
    """Conta sessões por split e rótulo."""

    counter = Counter((item.split, item.label) for item in assignments)
    return {
        split: {
            label: counter[(split, label)]
            for label in (LABEL_NORMAL, LABEL_ANOMALY)
        }
        for split in SPLIT_ORDER
    }

def evaluate_acceptance(
    source_labels: Mapping[str, str],
    assignments: Sequence[SplitAssignment],
    config: SplitConfig,
) -> dict[str, Any]:
    """Avalia invariantes sem corrigir silenciosamente os dados."""

    assigned_ids = [item.block_id for item in assignments]
    assigned_map = {item.block_id: item for item in assignments}
    domains_valid = all(
        item.label in EXPECTED_LABEL_COUNTS and item.split in SPLIT_ORDER
        for item in assignments
    )
    labels_preserved = all(
        block_id in assigned_map
        and assigned_map[block_id].label == label
        for block_id, label in source_labels.items()
    )
    actual = count_assignments(assignments)
    expected = expected_counts(config)
    train_normal_count = actual[SPLIT_TRAIN][LABEL_NORMAL]

    checks = {
        "expected_source_sessions": len(source_labels) == 575_061,
        "expected_source_normal": (
            Counter(source_labels.values())[LABEL_NORMAL] == 558_223
        ),
        "expected_source_anomaly": (
            Counter(source_labels.values())[LABEL_ANOMALY] == 16_838
        ),
        "one_assignment_per_session": len(assigned_ids) == len(set(assigned_ids)),
        "all_source_sessions_assigned": set(assigned_ids) == set(source_labels),
        "labels_preserved": labels_preserved,
        "domains_valid": domains_valid,
        "expected_counts_by_split_and_label": actual == expected,
        "train_normal_count": train_normal_count == 390_756,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "counts": actual,
    }

def load_accepted_audit(
    audit_path: Path,
    labels_path: Path,
) -> tuple[dict[str, any], str, str]:
    """Valida o aceite anterior e a identidade do arquivo de rótulos."""

    if not audit_path.is_file():
        raise FileNotFoundError(f"Manifesto de auditoria não encontrado: {audit_path}")

    with audit_path.open("r", encoding="utf-8") as stream:
        audit = json.load(stream)

    acceptance = audit.get("acceptance", {})
    if acceptance.get("accepted") is not True:
        raise ValueError("A auditoria da Etapa 1 não está aprovada.")
    checks = acceptance.get("checks", {})
    if not checks or any(value is not True for value in checks.values()):
        raise ValueError("A auditoria não contém todos os checks aprovados.")

    current_labels_sha256 = sha256_file(labels_path)
    audited_labels_sha256 = audit.get("inputs", {}).get("labels", {}).get("sha256")
    if not (
        current_labels_sha256
        == audited_labels_sha256
        == EXPECTED_LABELS_SHA256
    ):
        raise ValueError(
            "O checksum de anomaly_label.csv diverge da auditoria aprovada."
        )

    audited_log_sha256 = audit.get("inputs", {}).get("log", {}).get("sha256")
    if audited_log_sha256 != EXPECTED_LOG_SHA256:
        raise ValueError("O checksum aceito de HDFS.log não é o esperado.")

    return audit, sha256_file(audit_path), current_labels_sha256

def commit_temporary_file(temporary_path: Path, output_path: Path) -> None:
    """Publica um arquivo novo ou aceita um artefato existente idêntico."""

    if output_path.exists():
        if sha256_file(temporary_path) == sha256_file(output_path):
            temporary_path.unlink()
            return
        temporary_path.unlink()
        raise FileExistsError(
            f"Artefato divergente já existe e não será sobrescrito: {output_path}"
        )
    temporary_path.replace(output_path)


def write_split_csv(
    assignments: Sequence[SplitAssignment],
    output_path: Path,
) -> str:
    """Grava o CSV mestre em UTF-8, com ordem e quebras de linha estáveis."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("BlockId", "Label", "Split"))
        for item in sorted(assignments, key=lambda value: value.block_id):
            writer.writerow((item.block_id, item.label, item.split))
    commit_temporary_file(temporary_path, output_path)
    return sha256_file(output_path)


def write_train_normal(
    assignments: Sequence[SplitAssignment],
    output_path: Path,
) -> str:
    """Grava a visão regenerável de BlockIds normais do treino."""

    block_ids = sorted(
        item.block_id
        for item in assignments
        if item.split == SPLIT_TRAIN and item.label == LABEL_NORMAL
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        for block_id in block_ids:
            stream.write(f"{block_id}\n")
    commit_temporary_file(temporary_path, output_path)
    return sha256_file(output_path)


def write_json(payload: Mapping[str, Any], output_path: Path) -> str:
    """Grava JSON determinístico sem timestamp variável."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    commit_temporary_file(temporary_path, output_path)
    return sha256_file(output_path)

def decimal_ratios(config: SplitConfig) -> dict[str, str]:
    return {name: str(value) for name, value in config.ratios.items()}


def build_manifest(
    *,
    config: SplitConfig,
    audit_path: Path,
    audit_sha256: str,
    labels_path: Path,
    labels_sha256: str,
    split_path: Path,
    split_sha256: str,
    train_normal_path: Path,
    train_normal_sha256: str,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    """Registra a proveniência e o resultado lógico da Etapa 2."""

    return {
        "manifest_version": 1,
        "dataset": {"name": "Loghub HDFS_v1"},
        "upstream_audit": {
            "path": audit_path.as_posix(),
            "sha256": audit_sha256,
            "accepted": True,
            "accepted_log_sha256": EXPECTED_LOG_SHA256,
        },
        "input": {
            "labels_path": labels_path.as_posix(),
            "labels_sha256": labels_sha256,
            "sessions": 575_061,
        },
        "algorithm": {
            "name": config.algorithm,
            "version": config.algorithm_version,
            "salt": config.salt,
            "ratios": decimal_ratios(config),
            "rounding": "largest_remainder",
            "tie_break_order": list(SPLIT_ORDER),
            "assignment_inputs": ["BlockId", "Label"],
        },
        "model_usage_contract": {
            "master_train_keeps_anomalies": True,
            "unsupervised_fit_view": "split=train AND label=Normal",
            "validation_use": "hyperparameter_and_threshold_selection",
            "test_use": "final_evaluation_only",
        },
        "future_parser_contract": {
            "fit_partition": "train_normal",
            "validation_and_test_mode": "frozen_match_only",
            "unknown_template_token": "UNK_TEMPLATE",
            "parser_executed_in_this_stage": False,
        },
        "outputs": {
            "master_csv": {
                "path": split_path.as_posix(),
                "sha256": split_sha256,
                "data_rows": 575_061,
                "total_lines": 575_062,
                "size_bytes": split_path.stat().st_size,
            },
            "train_normal": {
                "path": train_normal_path.as_posix(),
                "sha256": train_normal_sha256,
                "total_lines": 390_756,
                "size_bytes": train_normal_path.stat().st_size,
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "acceptance": dict(acceptance),
    }

def create_split_artifacts(
    *,
    labels_path: Path,
    audit_path: Path,
    split_path: Path,
    train_normal_path: Path,
    manifest_path: Path,
    config: SplitConfig,
) -> dict[str, Any]:
    """Valida entradas, cria o split e grava seus três artefatos."""

    destinations = {
        path.resolve()
        for path in (split_path, train_normal_path, manifest_path)
    }
    if len(destinations) != 3:
        raise ValueError("Os três caminhos de saída devem ser distintos.")
    
    inputs = {labels_path.resolve(), audit_path.resolve()}

    if destinations.intersection(inputs):
        raise ValueError("Um caminho de saída não pode sobrescrever uma entrada.")

    _, audit_sha256, labels_sha256 = load_accepted_audit(
        audit_path,
        labels_path,
    )

    labels, label_metrics = load_labels(labels_path)

    if label_metrics["blank_values"] != 0:
        raise ValueError("O CSV possui valores vazios.")
    if label_metrics["invalid_block_ids"] != 0:
        raise ValueError("O CSV possui BlockIds inválidos.")
    if label_metrics["invalid_labels"] != 0:
        raise ValueError("O CSV possui rótulos inválidos.")
    if label_metrics["duplicate_rows"] != 0:
        raise ValueError("O CSV possui BlockIds duplicados.")
    if label_metrics["conflicting_rows"] != 0:
        raise ValueError("O CSV possui rótulos conflitantes.")

    assignments = create_stratified_split(labels, config)
    acceptance = evaluate_acceptance(labels, assignments, config)

    if acceptance["accepted"] is not True:
        failed = [
            name
            for name, passed in acceptance["checks"].items()
            if not passed
        ]
        raise ValueError(f"Split reprovado antes da escrita: {failed}")

    split_sha256 = write_split_csv(assignments, split_path)
    train_normal_sha256 = write_train_normal(assignments, train_normal_path)
    manifest = build_manifest(
        config=config,
        audit_path=audit_path,
        audit_sha256=audit_sha256,
        labels_path=labels_path,
        labels_sha256=labels_sha256,
        split_path=split_path,
        split_sha256=split_sha256,
        train_normal_path=train_normal_path,
        train_normal_sha256=train_normal_sha256,
        acceptance=acceptance,
    )
    write_json(manifest, manifest_path)
    return manifest

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cria o split mestre persistente do HDFS_v1."
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normal-train-output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-ratio", default="0.70")
    parser.add_argument("--validation-ratio", default="0.15")
    parser.add_argument("--test-ratio", default="0.15")
    parser.add_argument("--salt", default=DEFAULT_SALT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        config = SplitConfig(
            train_ratio=Decimal(arguments.train_ratio),
            validation_ratio=Decimal(arguments.validation_ratio),
            test_ratio=Decimal(arguments.test_ratio),
            salt=arguments.salt,
        )
        manifest = create_split_artifacts(
            labels_path=arguments.labels,
            audit_path=arguments.audit_manifest,
            split_path=arguments.output,
            train_normal_path=arguments.normal_train_output,
            manifest_path=arguments.manifest,
            config=config,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        InvalidOperation,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 2

    print("Split HDFS_v1 concluído.")
    print(f"Aceite: {'APROVADO' if manifest['acceptance']['accepted'] else 'REPROVADO'}")
    for split, labels in manifest["acceptance"]["counts"].items():
        print(
            f"{split}: Normal={labels['Normal']}, "
            f"Anomaly={labels['Anomaly']}"
        )
    return 0 if manifest["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())