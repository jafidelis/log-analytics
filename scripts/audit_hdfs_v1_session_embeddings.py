"""Audita o alinhamento das representações semânticas por sessão.

O script é somente leitura para corpus, índice e matrizes. Ele grava apenas
um manifesto de auditoria; não lê rótulos e não executa modelos.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED = {
    "train": {"sessions": 402_542, "events": 7_822_762, "unk": 3_845},
    "validation": {
        "sessions": 86_260,
        "events": 1_676_209,
        "unk": 852,
    },
    "test": {"sessions": 86_259, "events": 1_676_658, "unk": 787},
}
DIMENSION = 384
NORM_TOLERANCE = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON inválido: {path}")
    return value


def load_accepted_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    acceptance = manifest.get("acceptance")
    checks = acceptance.get("checks") if isinstance(acceptance, dict) else None
    if (
        not isinstance(acceptance, dict)
        or acceptance.get("accepted") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError(f"Manifesto não aprovado: {path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita índice, corpus e matrizes semânticas por sessão."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def audit_matrix(path: Path, expected_rows: int) -> dict[str, Any]:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (expected_rows, DIMENSION):
        raise ValueError(f"Shape inválido em {path}: {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"dtype inválido em {path}: {matrix.dtype}")

    min_norm = float("inf")
    max_norm = float("-inf")
    max_error = 0.0
    finite = True
    for start in range(0, expected_rows, 8192):
        block = np.asarray(matrix[start : start + 8192])
        finite = finite and bool(np.isfinite(block).all())
        norms = np.linalg.norm(block, axis=1)
        min_norm = min(min_norm, float(norms.min()))
        max_norm = max(max_norm, float(norms.max()))
        max_error = max(max_error, float(np.max(np.abs(norms - 1.0))))

    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "finite": finite,
        "norm_min": min_norm,
        "norm_max": max_norm,
        "norm_error_max": max_error,
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve(strict=True)
    rep = root / "artifacts/representation"

    corpus = rep / "hdfs_v1_session_corpus.csv.gz"
    corpus_manifest_path = rep / "hdfs_v1_session_corpus_manifest.json"
    counts_manifest_path = rep / "hdfs_v1_counts_manifest.json"
    sequences_manifest_path = rep / "hdfs_v1_sequences_manifest.json"
    session_manifest_path = (
        rep / "hdfs_v1_semantic_session_embedding_manifest.json"
    )
    counts = rep / "hdfs_v1_session_counts.csv.gz"
    sequences = rep / "hdfs_v1_session_sequences.csv.gz"
    index = rep / "hdfs_v1_semantic_session_index.csv.gz"
    output_manifest_path = (
        rep / "hdfs_v1_semantic_session_embedding_audit_manifest.json"
    )

    if output_manifest_path.exists():
        raise SystemExit(f"Artefato já existe: {output_manifest_path}")

    corpus_manifest = load_accepted_manifest(corpus_manifest_path)
    counts_manifest = load_accepted_manifest(counts_manifest_path)
    sequences_manifest = load_accepted_manifest(sequences_manifest_path)
    session_manifest = load_accepted_manifest(session_manifest_path)
    checks: dict[str, bool] = {
        "corpus_sha256_matches_manifest": sha256_file(corpus)
        == corpus_manifest["outputs"]["session_corpus_csv_gz"]["sha256"],
        "counts_sha256_matches_manifest": sha256_file(counts)
        == counts_manifest["outputs"]["session_counts_csv_gz"]["sha256"],
        "sequences_sha256_matches_manifest": sha256_file(sequences)
        == sequences_manifest["outputs"]["session_sequences_csv_gz"]["sha256"],
        "index_sha256_matches_manifest": sha256_file(index)
        == session_manifest["outputs"]["session_index"]["sha256"],
    }
    if not checks["corpus_sha256_matches_manifest"] or not checks[
        "index_sha256_matches_manifest"
    ]:
        raise SystemExit(f"Entradas divergentes: {checks}")

    matrix_paths = {
        split: rep / f"hdfs_v1_semantic_session_embeddings_{split}.npy"
        for split in EXPECTED
    }
    matrix_results = {
        split: audit_matrix(path, EXPECTED[split]["sessions"])
        for split, path in matrix_paths.items()
    }
    checks.update(
        {
            f"{split}_matrix_sha256_matches_manifest": sha256_file(path)
            == session_manifest["outputs"]["session_embeddings"][split][
                "sha256"
            ]
            for split, path in matrix_paths.items()
        }
    )

    counts_by_split = {split: 0 for split in EXPECTED}
    events_by_split = {split: 0 for split in EXPECTED}
    unk_by_split = {split: 0 for split in EXPECTED}
    seen_blocks: set[str] = set()
    previous_block = ""

    with (
        gzip.open(index, "rt", encoding="utf-8", newline="") as index_stream,
        gzip.open(corpus, "rt", encoding="utf-8", newline="") as corpus_stream,
        gzip.open(counts, "rt", encoding="utf-8", newline="") as counts_stream,
        gzip.open(sequences, "rt", encoding="utf-8", newline="") as sequence_stream,
    ):
        index_reader = csv.DictReader(index_stream)
        corpus_reader = csv.DictReader(corpus_stream)
        counts_reader = csv.DictReader(counts_stream)
        sequence_reader = csv.DictReader(sequence_stream)
        expected_index_header = [
            "Row",
            "BlockId",
            "Split",
            "Length",
            "UnkEventCount",
            "DocumentHash",
            "EmbeddingNorm",
        ]
        if index_reader.fieldnames != expected_index_header:
            raise ValueError("Cabeçalho inválido no índice")
        if corpus_reader.fieldnames != [
            "BlockId",
            "Split",
            "Length",
            "Document",
            "DocumentHash",
        ]:
            raise ValueError("Cabeçalho inválido no corpus")
        if counts_reader.fieldnames is None or counts_reader.fieldnames[:3] != [
            "BlockId",
            "Split",
            "Length",
        ] or counts_reader.fieldnames[-1] != "unk":
            raise ValueError("Cabeçalho inválido nas contagens")
        if sequence_reader.fieldnames != [
            "BlockId",
            "Split",
            "Length",
            "Sequence",
        ]:
            raise ValueError("Cabeçalho inválido nas sequências")

        for index_row, corpus_row, count_row, sequence_row in zip(
            index_reader,
            corpus_reader,
            counts_reader,
            sequence_reader,
        ):
            block_id = index_row["BlockId"]
            split = index_row["Split"]
            length = int(index_row["Length"])
            if split not in EXPECTED:
                raise ValueError(f"Split inválido: {split}")
            if block_id <= previous_block or block_id in seen_blocks:
                raise ValueError(f"BlockId duplicado/fora de ordem: {block_id}")
            previous_block = block_id
            seen_blocks.add(block_id)
            row = int(index_row["Row"])
            if row != counts_by_split[split]:
                raise ValueError(f"Row não contíguo em {split}: {row}")

            if any(
                (
                    block_id != corpus_row["BlockId"],
                    split != corpus_row["Split"],
                    length != int(corpus_row["Length"]),
                    corpus_row["DocumentHash"]
                    != hash_text(corpus_row["Document"]),
                    index_row["DocumentHash"] != corpus_row["DocumentHash"],
                    length != int(count_row["Length"]),
                    length != int(sequence_row["Length"]),
                    block_id != count_row["BlockId"],
                    split != count_row["Split"],
                    block_id != sequence_row["BlockId"],
                    split != sequence_row["Split"],
                    int(index_row["UnkEventCount"]) != int(count_row["unk"]),
                    sum(
                        int(count_row[f"t{i}"])
                        for i in range(1, 20)
                    )
                    + int(count_row["unk"])
                    != length,
                    len(sequence_row["Sequence"].split()) != length,
                    any(
                        int(value) < 0 or int(value) > 19
                        for value in sequence_row["Sequence"].split()
                    ),
                    sequence_row["Sequence"].split().count("0")
                    != int(index_row["UnkEventCount"]),
                )
            ):
                raise ValueError(f"Metadados divergentes: {block_id}")

            counts_by_split[split] += 1
            events_by_split[split] += length
            unk_by_split[split] += int(index_row["UnkEventCount"])

        if any(reader is not None for reader in (index_reader, corpus_reader)):
            if any(
                next(reader, None) is not None
                for reader in (
                    index_reader,
                    corpus_reader,
                    counts_reader,
                    sequence_reader,
                )
            ):
                raise ValueError("Artefatos possuem quantidades diferentes")

    checks.update(
        {
            "sessions_total_expected": sum(counts_by_split.values()) == 575_061,
            "events_total_expected": sum(events_by_split.values()) == 11_175_629,
            "unique_block_ids_expected": len(seen_blocks) == 575_061,
            "sessions_by_split_expected": counts_by_split == {
                split: values["sessions"] for split, values in EXPECTED.items()
            },
            "events_by_split_expected": events_by_split == {
                split: values["events"] for split, values in EXPECTED.items()
            },
            "unk_by_split_expected": unk_by_split == {
                split: values["unk"] for split, values in EXPECTED.items()
            },
            "matrices_finite": all(
                result["finite"] for result in matrix_results.values()
            ),
            "matrices_norms_expected": all(
                result["norm_error_max"] <= NORM_TOLERANCE
                for result in matrix_results.values()
            ),
        }
    )
    accepted = all(checks.values())
    manifest = {
        "stage": "etapa4b_auditoria_integridade_embeddings_sessao",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "inputs": {
            "corpus_sha256": sha256_file(corpus),
            "counts_sha256": sha256_file(counts),
            "sequences_sha256": sha256_file(sequences),
            "index_sha256": sha256_file(index),
            "session_embedding_manifest_sha256": sha256_file(session_manifest_path),
        },
        "results": {
            "sessions_by_split": counts_by_split,
            "events_by_split": events_by_split,
            "unk_by_split": unk_by_split,
            "unique_block_ids": len(seen_blocks),
            "matrices": matrix_results,
        },
        "acceptance": {"checks": checks, "accepted": accepted},
    }
    output_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not accepted:
        raise SystemExit(f"Auditoria reprovada: {checks}")
    print(
        json.dumps(
            {"accepted": accepted, "results": manifest["results"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
