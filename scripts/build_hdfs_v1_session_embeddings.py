"""Agrega embeddings de eventos em uma representação por sessão.

Usa os vetores congelados do catálogo semântico já aceito. Não lê rótulos,
não ajusta o encoder e não treina detector.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_DIMENSION = 384
EXPECTED_TOTAL_SESSIONS = 575_061
EXPECTED_TOTAL_EVENTS = 11_175_629
EXPECTED_SESSIONS = {
    "train": 402_542,
    "validation": 86_260,
    "test": 86_259,
}
EXPECTED_EVENTS = {
    "train": 7_822_762,
    "validation": 1_676_209,
    "test": 1_676_658,
}
CORPUS_HEADER = [
    "BlockId",
    "Split",
    "Length",
    "Document",
    "DocumentHash",
]
CATALOG_HEADER = [
    "CatalogIndex",
    "SourceType",
    "SourceId",
    "Text",
    "TextHash",
    "Occurrences",
]
INDEX_HEADER = [
    "Row",
    "BlockId",
    "Split",
    "Length",
    "UnkEventCount",
    "DocumentHash",
    "EmbeddingNorm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrega embeddings do catálogo semântico por sessão."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


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


def load_catalog(
    catalog_path: Path,
    vectors_path: Path,
) -> tuple[dict[str, np.ndarray], set[str], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with gzip.open(catalog_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CATALOG_HEADER:
            raise ValueError("Cabeçalho inválido no catálogo")
        for expected_index, row in enumerate(reader):
            if int(row["CatalogIndex"]) != expected_index:
                raise ValueError("CatalogIndex não contíguo")
            text = row["Text"]
            if not text or hash_text(text) != row["TextHash"]:
                raise ValueError("Texto/hash inválido no catálogo")
            entries.append(row)

    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (len(entries), EXPECTED_DIMENSION):
        raise ValueError(f"Shape do catálogo incompatível: {vectors.shape}")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Vetores do catálogo inválidos")

    by_hash: dict[str, np.ndarray] = {}
    unknown_hashes: set[str] = set()
    occurrences = 0
    for row in entries:
        text_hash = row["TextHash"]
        by_hash[text_hash] = np.asarray(
            vectors[int(row["CatalogIndex"])],
            dtype=np.float32,
        )
        occurrences += int(row["Occurrences"])
        if row["SourceType"] == "unk_prepared":
            unknown_hashes.add(text_hash)

    return by_hash, unknown_hashes, {
        "rows": len(entries),
        "occurrences": occurrences,
        "unknown_texts": len(unknown_hashes),
    }


def write_index_row(
    writer: csv.writer,
    row_number: int,
    block_id: str,
    split: str,
    length: int,
    unk_count: int,
    document_hash: str,
    embedding_norm: float,
) -> None:
    writer.writerow(
        [
            row_number,
            block_id,
            split,
            length,
            unk_count,
            document_hash,
            f"{embedding_norm:.9f}",
        ]
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve(strict=True)
    representation_dir = project_root / "artifacts/representation"

    corpus_path = representation_dir / "hdfs_v1_session_corpus.csv.gz"
    corpus_manifest_path = (
        representation_dir / "hdfs_v1_session_corpus_manifest.json"
    )
    gate_manifest_path = (
        representation_dir / "hdfs_v1_semantic_embedding_manifest.json"
    )
    catalog_path = representation_dir / "hdfs_v1_semantic_event_catalog.csv.gz"
    vectors_path = representation_dir / "hdfs_v1_semantic_catalog_embeddings.npy"

    output_paths = {
        split: representation_dir / f"hdfs_v1_semantic_session_embeddings_{split}.npy"
        for split in EXPECTED_SESSIONS
    }
    index_path = representation_dir / "hdfs_v1_semantic_session_index.csv.gz"
    manifest_path = (
        representation_dir
        / "hdfs_v1_semantic_session_embedding_manifest.json"
    )
    outputs = [*output_paths.values(), index_path, manifest_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    corpus_manifest = load_accepted_manifest(corpus_manifest_path)
    embedding_manifest = load_accepted_manifest(gate_manifest_path)
    if sha256_file(corpus_path) != corpus_manifest["outputs"][
        "session_corpus_csv_gz"
    ]["sha256"]:
        raise SystemExit("Corpus diverge do manifesto")
    if sha256_file(catalog_path) != embedding_manifest["inputs"][
        "catalog_sha256"
    ]:
        raise SystemExit("Catálogo diverge do manifesto de embeddings")
    if sha256_file(vectors_path) != embedding_manifest["outputs"][
        "catalog_embeddings_npy"
    ]["sha256"]:
        raise SystemExit("Vetores do catálogo divergem do manifesto")

    catalog_by_hash, unknown_hashes, catalog_summary = load_catalog(
        catalog_path,
        vectors_path,
    )

    temp_paths = {
        split: Path(f"{path}.tmp") for split, path in output_paths.items()
    }
    index_tmp = Path(f"{index_path}.tmp")
    manifest_tmp = Path(f"{manifest_path}.tmp")
    matrices = {
        split: np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(EXPECTED_SESSIONS[split], EXPECTED_DIMENSION),
        )
        for split, path in temp_paths.items()
    }
    rows_by_split: Counter[str] = Counter()
    events_by_split: Counter[str] = Counter()
    unknown_by_split: Counter[str] = Counter()
    unresolved_events = 0
    malformed_documents = 0
    nonfinite_sessions = 0
    zero_norm_sessions = 0
    min_session_norm = float("inf")
    max_session_norm = float("-inf")
    max_norm_error = 0.0
    previous_block_id = ""

    with index_tmp.open("wb") as raw_index:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_index,
            mode="wb",
            mtime=0,
        ) as gzip_index:
            with io.TextIOWrapper(
                gzip_index,
                encoding="utf-8",
                newline="",
            ) as index_stream:
                writer = csv.writer(index_stream)
                writer.writerow(INDEX_HEADER)

                with gzip.open(
                    corpus_path,
                    "rt",
                    encoding="utf-8",
                    newline="",
                ) as corpus_stream:
                    reader = csv.DictReader(corpus_stream)
                    if reader.fieldnames != CORPUS_HEADER:
                        raise ValueError("Cabeçalho inválido no corpus")

                    for row in reader:
                        block_id = row["BlockId"]
                        split = row["Split"]
                        if block_id <= previous_block_id:
                            raise ValueError("Corpus fora de ordem ou duplicado")
                        previous_block_id = block_id
                        if split not in EXPECTED_SESSIONS:
                            raise ValueError(f"Split inválido: {split}")

                        length = int(row["Length"])
                        document = row["Document"]
                        document_hash = row["DocumentHash"]
                        if hash_text(document) != document_hash:
                            raise ValueError(f"DocumentHash inválido: {block_id}")
                        event_texts = document.splitlines()
                        if len(event_texts) != length or not event_texts:
                            malformed_documents += 1
                            raise ValueError(f"Document inválido: {block_id}")

                        event_vectors: list[np.ndarray] = []
                        unk_count = 0
                        for event_text in event_texts:
                            text_hash = hash_text(event_text)
                            vector = catalog_by_hash.get(text_hash)
                            if vector is None:
                                unresolved_events += 1
                                raise ValueError(
                                    f"Evento sem vetor no catálogo: {block_id}"
                                )
                            event_vectors.append(vector)
                            unk_count += int(text_hash in unknown_hashes)

                        summed = np.sum(
                            np.stack(event_vectors),
                            axis=0,
                            dtype=np.float64,
                        ) / length
                        norm = float(np.linalg.norm(summed))
                        if not np.isfinite(norm) or norm == 0.0:
                            raise ValueError(f"Norma inválida: {block_id}")
                        session_vector = np.asarray(
                            summed / norm,
                            dtype=np.float32,
                        )
                        session_norm = float(
                            np.linalg.norm(session_vector)
                        )
                        if not np.isfinite(session_vector).all():
                            nonfinite_sessions += 1
                            raise ValueError(
                                f"Vetor de sessão não finito: {block_id}"
                            )
                        if not np.isfinite(session_norm):
                            nonfinite_sessions += 1
                            raise ValueError(
                                f"Norma de sessão não finita: {block_id}"
                            )
                        if session_norm == 0.0:
                            zero_norm_sessions += 1
                            raise ValueError(
                                f"Norma de sessão zero: {block_id}"
                            )
                        min_session_norm = min(
                            min_session_norm,
                            session_norm,
                        )
                        max_session_norm = max(
                            max_session_norm,
                            session_norm,
                        )
                        max_norm_error = max(
                            max_norm_error,
                            abs(session_norm - 1.0),
                        )

                        row_number = rows_by_split[split]
                        matrices[split][row_number] = session_vector
                        rows_by_split[split] += 1
                        events_by_split[split] += length
                        unknown_by_split[split] += unk_count
                        write_index_row(
                            writer,
                            row_number,
                            block_id,
                            split,
                            length,
                            unk_count,
                            document_hash,
                            session_norm,
                        )

    for matrix in matrices.values():
        matrix.flush()
        del matrix

    checks = {
        "catalog_rows_expected": catalog_summary["rows"] == 285,
        "catalog_occurrences_expected": catalog_summary["occurrences"]
        == EXPECTED_TOTAL_EVENTS,
        "all_events_resolved": unresolved_events == 0,
        "documents_well_formed": malformed_documents == 0,
        "sessions_total_expected": sum(rows_by_split.values())
        == EXPECTED_TOTAL_SESSIONS,
        "events_total_expected": sum(events_by_split.values())
        == EXPECTED_TOTAL_EVENTS,
        "sessions_by_split_expected": dict(rows_by_split) == EXPECTED_SESSIONS,
        "events_by_split_expected": dict(events_by_split) == EXPECTED_EVENTS,
        "session_vectors_finite": nonfinite_sessions == 0,
        "session_vectors_nonzero": zero_norm_sessions == 0,
        "session_norms_expected": max_norm_error <= 1e-5,
    }
    accepted = all(checks.values())
    if not accepted:
        raise SystemExit(f"Agregação reprovada: {checks}")

    manifest = {
        "stage": "etapa4b_agregacao_semantica_por_sessao",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sentence_transformers": version("sentence-transformers"),
        },
        "representation": {
            "unit": "sessão BlockId",
            "event_embedding_source": "semantic catalog embeddings",
            "pooling": "arithmetic mean weighted by event repetition",
            "post_pooling_normalization": "L2",
            "labels_in_features": False,
            "sequence_preserved_elsewhere": True,
        },
        "inputs": {
            "corpus_sha256": sha256_file(corpus_path),
            "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "catalog_sha256": sha256_file(catalog_path),
            "catalog_embeddings_sha256": sha256_file(vectors_path),
            "embedding_manifest_sha256": sha256_file(gate_manifest_path),
        },
        "results": {
            "sessions_by_split": dict(rows_by_split),
            "events_by_split": dict(events_by_split),
            "unk_events_by_split": dict(unknown_by_split),
            "catalog_rows": catalog_summary["rows"],
            "embedding_dimension": EXPECTED_DIMENSION,
            "session_norm_min": min_session_norm,
            "session_norm_max": max_session_norm,
            "session_norm_error_max": max_norm_error,
            "unresolved_events": unresolved_events,
            "nonfinite_sessions": nonfinite_sessions,
            "zero_norm_sessions": zero_norm_sessions,
        },
        "outputs": {
            "session_embeddings": {
                split: {
                    "path": str(path.relative_to(project_root)),
                    "shape": [EXPECTED_SESSIONS[split], EXPECTED_DIMENSION],
                    "dtype": "float32",
                    "sha256": sha256_file(temp_paths[split]),
                    "size_bytes": temp_paths[split].stat().st_size,
                }
                for split, path in output_paths.items()
            },
            "session_index": {
                "path": str(index_path.relative_to(project_root)),
                "sha256": sha256_file(index_tmp),
                "size_bytes": index_tmp.stat().st_size,
            },
        },
        "acceptance": {
            "checks": checks,
            "accepted": accepted,
        },
    }
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for split, path in output_paths.items():
        os.replace(temp_paths[split], path)
    os.replace(index_tmp, index_path)
    os.replace(manifest_tmp, manifest_path)

    print(
        json.dumps(
            {
                "accepted": accepted,
                "sessions_by_split": dict(rows_by_split),
                "events_by_split": dict(events_by_split),
                "manifest": str(manifest_path.relative_to(project_root)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
