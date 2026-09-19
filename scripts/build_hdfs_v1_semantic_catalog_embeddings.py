"""Gera embeddings do catálogo textual e similaridades dos templates.

Este é o segundo incremento do gate semântico. Codifica somente os 285 textos
únicos já auditados; não lê rótulos, não agrega sessões e não treina detector.
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
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


EXPECTED_MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
EXPECTED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EXPECTED_TEXTS = 285
EXPECTED_TEMPLATES = 19
EXPECTED_EVENTS = 11_175_629
EXPECTED_DIMENSION = 384
EXPECTED_MAX_SEQ_LENGTH = 128
MAX_ABS_DIFFERENCE = 1e-6
NORM_TOLERANCE = 1e-5

CATALOG_HEADER = [
    "CatalogIndex",
    "SourceType",
    "SourceId",
    "Text",
    "TextHash",
    "Occurrences",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Codifica os textos únicos do catálogo HDFS v1 e "
            "calcula similaridades entre templates."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Raiz do projeto log-analytics.",
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


def resolve_under_root(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Caminho fora do projeto: {relative_path}") from error
    return path


def verify_model_files(
    project_root: Path,
    files_sha256: dict[str, str],
) -> bool:
    if not files_sha256:
        return False
    for relative_path, expected_sha256 in files_sha256.items():
        path = resolve_under_root(project_root, relative_path)
        if not path.is_file() or sha256_file(path) != expected_sha256:
            return False
    return True


def load_catalog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text_hashes: set[str] = set()
    texts: set[str] = set()

    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CATALOG_HEADER:
            raise ValueError(f"Cabeçalho inesperado: {path}")

        for expected_index, row in enumerate(reader):
            index = int(row["CatalogIndex"])
            if index != expected_index:
                raise ValueError(f"CatalogIndex inválido: {index}")

            text = row["Text"]
            text_hash = row["TextHash"]
            source_type = row["SourceType"]
            source_id = row["SourceId"]
            if not text or hash_text(text) != text_hash:
                raise ValueError(f"Texto/hash inválido no índice {index}")
            if text_hash in text_hashes or text in texts:
                raise ValueError(f"Duplicata no catálogo: {index}")

            occurrences = int(row["Occurrences"])
            if occurrences <= 0:
                raise ValueError(f"Ocorrência inválida no índice {index}")
            if source_type == "template":
                expected_id = str(sum(
                    value["SourceType"] == "template" for value in rows
                ) + 1)
                if source_id != expected_id:
                    raise ValueError(
                        f"SourceId de template inválido no índice {index}"
                    )
            elif source_type == "unk_prepared":
                if source_id != text_hash:
                    raise ValueError(
                        f"SourceId de UNK inválido no índice {index}"
                    )
            else:
                raise ValueError(f"SourceType inválido no índice {index}")

            text_hashes.add(text_hash)
            texts.add(text)
            rows.append(
                {
                    "CatalogIndex": index,
                    "SourceType": source_type,
                    "SourceId": source_id,
                    "Text": text,
                    "TextHash": text_hash,
                    "Occurrences": occurrences,
                }
            )

    if len(rows) != EXPECTED_TEXTS:
        raise ValueError(f"Esperados {EXPECTED_TEXTS} textos: {len(rows)}")
    if sum(row["Occurrences"] for row in rows) != EXPECTED_EVENTS:
        raise ValueError("Soma de ocorrências divergente do corpus")
    if sum(row["SourceType"] == "template" for row in rows) != EXPECTED_TEMPLATES:
        raise ValueError("Quantidade de templates divergente")
    if sum(row["SourceType"] == "unk_prepared" for row in rows) != (
        EXPECTED_TEXTS - EXPECTED_TEMPLATES
    ):
        raise ValueError("Quantidade de textos UNK divergente")
    return rows


def load_model(
    project_root: Path,
    model_manifest_path: Path,
) -> tuple[SentenceTransformer, dict[str, Any], Path]:
    manifest = load_json(model_manifest_path)
    if manifest.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError("Model_id divergente do contrato")
    if manifest.get("snapshot_revision") != EXPECTED_REVISION:
        raise ValueError("Revisão do modelo divergente do contrato")
    snapshot_path = resolve_under_root(
        project_root,
        manifest["snapshot_path"],
    )
    if snapshot_path.name != EXPECTED_REVISION:
        raise ValueError("Diretório do snapshot divergente")
    if not verify_model_files(
        project_root,
        manifest.get("files_sha256", {}),
    ):
        raise ValueError("Arquivos do snapshot divergem do manifesto")

    model = SentenceTransformer(
        str(snapshot_path),
        local_files_only=True,
        device="cpu",
    )
    model.eval()
    if model.get_embedding_dimension() != EXPECTED_DIMENSION:
        raise ValueError("Dimensão do encoder divergente")
    if int(model.max_seq_length) != EXPECTED_MAX_SEQ_LENGTH:
        raise ValueError("max_seq_length divergente")
    return model, manifest, snapshot_path


def encode_catalog(
    model: SentenceTransformer,
    texts: list[str],
) -> tuple[np.ndarray, float]:
    with torch.inference_mode():
        first = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device="cpu",
        )
        second = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cpu",
        )

    vectors = np.asarray(first, dtype=np.float32)
    repeated = np.asarray(second, dtype=np.float32)
    max_difference = float(np.max(np.abs(vectors - repeated)))

    if vectors.shape != (EXPECTED_TEXTS, EXPECTED_DIMENSION):
        raise ValueError(f"Shape inesperado: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding não finito")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
        raise ValueError("Normas dos embeddings fora da tolerância")
    if max_difference > MAX_ABS_DIFFERENCE:
        raise ValueError(
            f"Codificação não determinística: {max_difference}"
        )
    return vectors, max_difference


def write_similarity(
    path: Path,
    catalog: list[dict[str, Any]],
    vectors: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    template_rows = [
        (index, row)
        for index, row in enumerate(catalog)
        if row["SourceType"] == "template"
    ]
    if [row["SourceId"] for _, row in template_rows] != [
        str(value) for value in range(1, EXPECTED_TEMPLATES + 1)
    ]:
        raise ValueError("Templates não estão ordenados por TemplateId")

    template_vectors = vectors[[index for index, _ in template_rows]]
    similarity = template_vectors @ template_vectors.T
    if not np.allclose(similarity, similarity.T, atol=1e-6):
        raise ValueError("Matriz de similaridade não simétrica")
    if not np.allclose(np.diag(similarity), 1.0, atol=NORM_TOLERANCE):
        raise ValueError("Diagonal da similaridade inválida")

    matrix_header = ["TemplateId", "TextHash", *[f"t{i}" for i in range(1, 20)]]
    matrix_rows = []
    neighbors: list[dict[str, Any]] = []
    for row_index, (_, row) in enumerate(template_rows):
        matrix_rows.append(
            [
                row["SourceId"],
                row["TextHash"],
                *[f"{value:.9f}" for value in similarity[row_index]],
            ]
        )
        ranking = np.argsort(-similarity[row_index])
        rank = 0
        for neighbor_index in ranking:
            if neighbor_index == row_index:
                continue
            rank += 1
            neighbor = template_rows[int(neighbor_index)][1]
            neighbors.append(
                {
                    "TemplateId": row["SourceId"],
                    "Rank": rank,
                    "NeighborTemplateId": neighbor["SourceId"],
                    "Cosine": f"{similarity[row_index, neighbor_index]:.9f}",
                    "NeighborText": neighbor["Text"],
                }
            )
            if rank == 3:
                break

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(matrix_header)
        writer.writerows(matrix_rows)

    return similarity, neighbors


def write_neighbors(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "TemplateId",
        "Rank",
        "NeighborTemplateId",
        "Cosine",
        "NeighborText",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve(strict=True)
    representation_dir = project_root / "artifacts/representation"

    catalog_path = representation_dir / "hdfs_v1_semantic_event_catalog.csv.gz"
    gate_manifest_path = representation_dir / "hdfs_v1_semantic_gate_manifest.json"
    model_manifest_path = (
        project_root / "artifacts/models/embedding_model_manifest.json"
    )
    vectors_path = representation_dir / "hdfs_v1_semantic_catalog_embeddings.npy"
    similarity_path = representation_dir / "hdfs_v1_semantic_template_similarity.csv"
    neighbors_path = representation_dir / "hdfs_v1_semantic_template_neighbors.csv"
    manifest_path = representation_dir / "hdfs_v1_semantic_embedding_manifest.json"

    outputs = [vectors_path, similarity_path, neighbors_path, manifest_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    gate_manifest = load_accepted_manifest(gate_manifest_path)
    catalog_sha = sha256_file(catalog_path)
    expected_catalog_sha = gate_manifest["outputs"][
        "semantic_event_catalog_csv_gz"
    ]["sha256"]
    if catalog_sha != expected_catalog_sha:
        raise SystemExit("Catálogo diverge do manifesto do gate")

    catalog = load_catalog(catalog_path)
    model, model_manifest, snapshot_path = load_model(
        project_root,
        model_manifest_path,
    )
    texts = [row["Text"] for row in catalog]
    vectors, max_difference = encode_catalog(model, texts)

    vectors_tmp = Path(f"{vectors_path}.tmp")
    similarity_tmp = Path(f"{similarity_path}.tmp")
    neighbors_tmp = Path(f"{neighbors_path}.tmp")
    manifest_tmp = Path(f"{manifest_path}.tmp")

    with vectors_tmp.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    similarity, neighbors = write_similarity(
        similarity_tmp,
        catalog,
        vectors,
    )
    write_neighbors(neighbors_tmp, neighbors)

    norms = np.linalg.norm(vectors, axis=1)
    checks = {
        "gate_manifest_accepted": True,
        "catalog_rows_expected": len(catalog) == EXPECTED_TEXTS,
        "embedding_shape_expected": list(vectors.shape)
        == [EXPECTED_TEXTS, EXPECTED_DIMENSION],
        "embedding_dtype_float32": vectors.dtype == np.float32,
        "embeddings_finite": bool(np.isfinite(vectors).all()),
        "embedding_norms_expected": bool(
            np.allclose(norms, 1.0, atol=NORM_TOLERANCE)
        ),
        "encoding_deterministic": max_difference <= MAX_ABS_DIFFERENCE,
        "similarity_shape_expected": list(similarity.shape)
        == [EXPECTED_TEMPLATES, EXPECTED_TEMPLATES],
        "similarity_symmetric": bool(
            np.allclose(similarity, similarity.T, atol=1e-6)
        ),
        "similarity_diagonal_expected": bool(
            np.allclose(np.diag(similarity), 1.0, atol=NORM_TOLERANCE)
        ),
    }
    accepted = all(checks.values())
    if not accepted:
        raise SystemExit(f"Gate reprovado: {checks}")

    manifest = {
        "stage": "etapa4b_gate_semantico_embeddings_catalogo",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": version("torch"),
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
        },
        "model": {
            "model_id": EXPECTED_MODEL_ID,
            "snapshot_path": str(snapshot_path.relative_to(project_root)),
            "snapshot_revision": snapshot_path.name,
            "embedding_dimension": model.get_embedding_dimension(),
            "max_seq_length": int(model.max_seq_length),
            "device": "cpu",
            "batch_size": 32,
            "normalize_embeddings": True,
        },
        "inputs": {
            "catalog_sha256": catalog_sha,
            "gate_manifest_sha256": sha256_file(gate_manifest_path),
            "model_manifest_sha256": sha256_file(model_manifest_path),
        },
        "results": {
            "catalog_rows": len(catalog),
            "templates": EXPECTED_TEMPLATES,
            "embedding_shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "max_abs_difference": max_difference,
            "semantic_quality_review_required": True,
        },
        "outputs": {
            "catalog_embeddings_npy": {
                "path": str(vectors_path.relative_to(project_root)),
                "sha256": sha256_file(vectors_tmp),
                "size_bytes": vectors_tmp.stat().st_size,
            },
            "template_similarity_csv": {
                "path": str(similarity_path.relative_to(project_root)),
                "sha256": sha256_file(similarity_tmp),
                "size_bytes": similarity_tmp.stat().st_size,
            },
            "template_neighbors_csv": {
                "path": str(neighbors_path.relative_to(project_root)),
                "sha256": sha256_file(neighbors_tmp),
                "size_bytes": neighbors_tmp.stat().st_size,
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
    os.replace(vectors_tmp, vectors_path)
    os.replace(similarity_tmp, similarity_path)
    os.replace(neighbors_tmp, neighbors_path)
    os.replace(manifest_tmp, manifest_path)

    print(
        json.dumps(
            {
                "accepted": accepted,
                "shape": list(vectors.shape),
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
                "max_abs_difference": max_difference,
                "manifest": str(manifest_path.relative_to(project_root)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
