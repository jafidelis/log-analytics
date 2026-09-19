"""Carregamento validado das representações da Etapa 4.

Toda leitura passa por verificação de manifesto aprovado e checksum;
nenhum detector deve ler os artefatos por outro caminho.
"""

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from log_analytics.data.audit_hdfs_v1 import sha256_file

VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SessionCounts:
    """Vetores de contagem de uma partição, ordenados por BlockId."""
    split: str
    block_ids: list[str]
    features: list[list[int]]  # t1..t19 + unk, nesta ordem
    lengths: list[int]


def load_accepted_manifest(path: Path) -> dict:
    """Lê um manifesto e exige acceptance.accepted com todos os checks."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    acceptance = manifest["acceptance"]
    checks = acceptance.get("checks", {})

    if acceptance.get("accepted") is not True or not checks or any(
        value is not True for value in checks.values()
    ):
        raise ValueError(f"Manifesto não aprovado: {path}")

    return manifest


def load_train_normal_ids(
    list_path: Path,
    split_manifest_path: Path,
) -> frozenset[str]:
    """Carrega a lista oficial de sessões de treino normais (Etapa 2)."""
    manifest = load_accepted_manifest(split_manifest_path)
    expected = manifest["outputs"]["train_normal"]

    if sha256_file(list_path) != expected["sha256"]:
        raise ValueError("Lista train_normal diverge do manifesto.")

    ids = list_path.read_text(encoding="utf-8").splitlines()

    if len(ids) != expected["total_lines"]:
        raise ValueError("Contagem da lista train_normal inesperada.")
    if len(set(ids)) != len(ids):
        raise ValueError("Lista train_normal com BlockId duplicado.")

    return frozenset(ids)


def load_session_counts(
    counts_path: Path,
    counts_manifest_path: Path,
    split: str,
) -> SessionCounts:
    """Carrega os vetores de contagem de uma partição."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Partição inválida: {split}")

    manifest = load_accepted_manifest(counts_manifest_path)
    expected_sha256 = manifest["outputs"]["session_counts_csv_gz"]["sha256"]

    if sha256_file(counts_path) != expected_sha256:
        raise ValueError("Artefato de contagens diverge do manifesto.")

    columns = manifest["representation"]["columns"]
    feature_names = columns[3:]  # após BlockId, Split, Length

    block_ids: list[str] = []
    features: list[list[int]] = []
    lengths: list[int] = []
    previous_block_id = ""

    with gzip.open(
        counts_path, "rt", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.reader(stream)

        if next(reader) != columns:
            raise ValueError("Cabeçalho diverge do manifesto.")

        for row in reader:
            block_id, row_split, length = row[0], row[1], row[2]

            if block_id <= previous_block_id:
                raise ValueError(
                    f"Ordenação violada em {block_id}."
                )
            previous_block_id = block_id

            if row_split != split:
                continue

            vector = [int(value) for value in row[3:]]

            if sum(vector) != int(length):
                raise ValueError(
                    f"Contagens não somam Length em {block_id}."
                )

            block_ids.append(block_id)
            features.append(vector)
            lengths.append(int(length))

    if not block_ids:
        raise ValueError(f"Nenhuma sessão na partição {split}.")

    assert len(feature_names) == len(features[0])
    return SessionCounts(split, block_ids, features, lengths)


def select_train_normal(
    counts: SessionCounts,
    train_normal_ids: frozenset[str],
) -> SessionCounts:
    """Filtra a visão train_normal a partir da partição de treino."""
    if counts.split != "train":
        raise ValueError("train_normal deriva da partição train.")

    block_ids: list[str] = []
    features: list[list[int]] = []
    lengths: list[int] = []

    for block_id, vector, length in zip(
        counts.block_ids, counts.features, counts.lengths
    ):
        if block_id in train_normal_ids:
            block_ids.append(block_id)
            features.append(vector)
            lengths.append(length)

    if len(block_ids) != len(train_normal_ids):
        raise ValueError(
            "Sessões train_normal ausentes na partição de treino."
        )

    return SessionCounts("train_normal", block_ids, features, lengths)