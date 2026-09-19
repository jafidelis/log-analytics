"""Compara representações no mesmo protocolo OCSVM, somente na validação.

Condições: contagens, TF-IDF, embedding semântico e embedding + contagens.
O teste não é lido nem avaliado.
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
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from log_analytics.data.detection_metrics import sweep_best_threshold
from log_analytics.data.load_representation_hdfs_v1 import (
    load_session_counts,
    load_train_normal_ids,
)
from log_analytics.data.session_hdfs_v1 import load_split_index
from log_analytics.data.split_hdfs_v1 import LABEL_ANOMALY


SEED = 20260911
SUBSAMPLE_SIZE = 20_000
NU = 0.01
GAMMA = "scale"
EXPECTED_TRAIN_NORMAL = 390_756
EXPECTED_VALIDATION = 86_260
EXPECTED_VALIDATION_ANOMALIES = 2_526


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara representações do HDFS v1 na validação."
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


def load_semantic_split(
    matrix_path: Path,
    index_path: Path,
    split: str,
) -> tuple[list[str], np.ndarray]:
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    block_ids: list[str] = []
    rows: list[int] = []
    with gzip.open(index_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["Split"] != split:
                continue
            block_ids.append(row["BlockId"])
            rows.append(int(row["Row"]))
    if not rows or rows != list(range(len(rows))):
        raise ValueError(f"Índice semântico inválido: {split}")
    if matrix.shape[0] != len(rows) or matrix.shape[1] != 384:
        raise ValueError(f"Matriz semântica incompatível: {split}")
    return block_ids, np.asarray(matrix, dtype=np.float32)


def load_tfidf_split(
    matrix_path: Path,
    index_path: Path,
    split: str,
) -> tuple[list[str], np.ndarray]:
    matrix = load_npz(matrix_path).tocsr()
    block_ids: list[str] = []
    rows: list[int] = []
    with gzip.open(index_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["Split"] != split:
                continue
            block_ids.append(row["BlockId"])
            rows.append(int(row["Row"]))
    if rows != list(range(len(rows))) or matrix.shape[0] != len(rows):
        raise ValueError(f"Índice TF-IDF inválido: {split}")
    return block_ids, matrix.toarray().astype(np.float32, copy=False)


def evaluate_representation(
    name: str,
    train: np.ndarray,
    validation: np.ndarray,
    train_normal_positions: np.ndarray,
    y_validation: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(train[train_normal_positions])
    rng = np.random.default_rng(SEED)
    subsample = np.sort(
        rng.choice(
            train_normal_positions.shape[0],
            size=SUBSAMPLE_SIZE,
            replace=False,
        )
    )
    normal_train = train[train_normal_positions]
    x_fit = scaler.transform(normal_train[subsample])
    x_validation = scaler.transform(validation)

    model = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA)
    model.fit(x_fit)
    scores = -model.decision_function(x_validation)
    threshold = sweep_best_threshold(y_validation, scores)
    return {
        "representation": name,
        "dimensions": int(train.shape[1]),
        "fit_sessions": int(normal_train.shape[0]),
        "subsample_size": int(x_fit.shape[0]),
        "params": {"nu": NU, "gamma": GAMMA},
        "pr_auc": round(float(average_precision_score(y_validation, scores)), 4),
        "validation": threshold,
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve(strict=True)
    rep = root / "artifacts/representation"
    detection = root / "artifacts/detection"
    report_path = detection / "hdfs_v1_representation_comparison_validation.json"
    if report_path.exists():
        raise SystemExit(f"Relatório já existe: {report_path}")

    counts_path = rep / "hdfs_v1_session_counts.csv.gz"
    counts_manifest_path = rep / "hdfs_v1_counts_manifest.json"
    tfidf_manifest_path = rep / "hdfs_v1_tfidf_manifest.json"
    tfidf_index_path = rep / "hdfs_v1_tfidf_session_index.csv.gz"
    semantic_index_path = rep / "hdfs_v1_semantic_session_index.csv.gz"
    semantic_manifest_path = rep / "hdfs_v1_semantic_session_embedding_manifest.json"
    audit_manifest_path = rep / "hdfs_v1_semantic_session_embedding_audit_manifest.json"
    split_path = root / "artifacts/data_splits/hdfs_v1_session_split.csv"
    split_manifest_path = root / "artifacts/data_splits/hdfs_v1_split_manifest.json"
    train_normal_path = root / "artifacts/data_splits/hdfs_v1_train_normal_block_ids.txt"

    counts_manifest = load_accepted_manifest(counts_manifest_path)
    tfidf_manifest = load_accepted_manifest(tfidf_manifest_path)
    semantic_manifest = load_accepted_manifest(semantic_manifest_path)
    load_accepted_manifest(audit_manifest_path)
    split_manifest = load_accepted_manifest(split_manifest_path)

    checks = {
        "counts_sha256_matches_manifest": sha256_file(counts_path)
        == counts_manifest["outputs"]["session_counts_csv_gz"]["sha256"],
        "split_sha256_matches_manifest": sha256_file(split_path)
        == split_manifest["outputs"]["master_csv"]["sha256"],
        "tfidf_train_sha256_matches_manifest": sha256_file(rep / "hdfs_v1_tfidf_train.npz")
        == tfidf_manifest["outputs"]["matrices"]["train"]["sha256"],
        "tfidf_validation_sha256_matches_manifest": sha256_file(
            rep / "hdfs_v1_tfidf_validation.npz"
        )
        == tfidf_manifest["outputs"]["matrices"]["validation"]["sha256"],
        "tfidf_index_sha256_matches_manifest": sha256_file(
            tfidf_index_path
        )
        == tfidf_manifest["outputs"]["session_index"]["sha256"],
        "semantic_index_sha256_matches_manifest": sha256_file(
            semantic_index_path
        )
        == semantic_manifest["outputs"]["session_index"]["sha256"],
        "semantic_train_sha256_matches_manifest": sha256_file(
            rep / "hdfs_v1_semantic_session_embeddings_train.npy"
        )
        == semantic_manifest["outputs"]["session_embeddings"]["train"][
            "sha256"
        ],
        "semantic_validation_sha256_matches_manifest": sha256_file(
            rep / "hdfs_v1_semantic_session_embeddings_validation.npy"
        )
        == semantic_manifest["outputs"]["session_embeddings"]["validation"][
            "sha256"
        ],
        "semantic_manifest_not_tested": semantic_manifest["representation"].get(
            "labels_in_features", False
        ) is False,
    }
    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    train_counts = load_session_counts(
        counts_path, counts_manifest_path, "train"
    )
    validation_counts = load_session_counts(
        counts_path, counts_manifest_path, "validation"
    )
    train_normal_ids = load_train_normal_ids(
        train_normal_path, split_manifest_path
    )
    split_index = load_split_index(split_path)
    y_validation = np.asarray(
        [
            split_index[block_id].label == LABEL_ANOMALY
            for block_id in validation_counts.block_ids
        ],
        dtype=bool,
    )

    train_counts_array = np.asarray(train_counts.features, dtype=np.float32)
    validation_counts_array = np.asarray(
        validation_counts.features, dtype=np.float32
    )
    train_normal_positions = np.asarray(
        [
            index
            for index, block_id in enumerate(train_counts.block_ids)
            if block_id in train_normal_ids
        ],
        dtype=np.int64,
    )
    if (
        train_normal_positions.size != EXPECTED_TRAIN_NORMAL
        or validation_counts_array.shape[0] != EXPECTED_VALIDATION
        or int(y_validation.sum()) != EXPECTED_VALIDATION_ANOMALIES
    ):
        raise SystemExit("Cardinalidades de treino/validação inválidas")

    semantic_train_ids, semantic_train = load_semantic_split(
        rep / "hdfs_v1_semantic_session_embeddings_train.npy",
        semantic_index_path,
        "train",
    )
    semantic_validation_ids, semantic_validation = load_semantic_split(
        rep / "hdfs_v1_semantic_session_embeddings_validation.npy",
        semantic_index_path,
        "validation",
    )
    if (
        semantic_train_ids != train_counts.block_ids
        or semantic_validation_ids != validation_counts.block_ids
    ):
        raise ValueError("Alinhamento semântico diverge das contagens")
    checks["semantic_alignment_matches_counts"] = True

    results_by_name = {}
    results_by_name["counts"] = evaluate_representation(
        "counts",
        train_counts_array,
        validation_counts_array,
        train_normal_positions,
        y_validation,
    )
    results_by_name["semantic"] = evaluate_representation(
        "semantic",
        semantic_train,
        semantic_validation,
        train_normal_positions,
        y_validation,
    )

    semantic_plus_train = np.concatenate(
        (semantic_train, train_counts_array), axis=1
    )
    semantic_plus_validation = np.concatenate(
        (semantic_validation, validation_counts_array), axis=1
    )
    del semantic_train, semantic_validation
    results_by_name["semantic_plus_counts"] = evaluate_representation(
        "semantic_plus_counts",
        semantic_plus_train,
        semantic_plus_validation,
        train_normal_positions,
        y_validation,
    )
    del semantic_plus_train, semantic_plus_validation

    tfidf_train_ids, tfidf_train = load_tfidf_split(
        rep / "hdfs_v1_tfidf_train.npz", tfidf_index_path, "train"
    )
    tfidf_validation_ids, tfidf_validation = load_tfidf_split(
        rep / "hdfs_v1_tfidf_validation.npz", tfidf_index_path, "validation"
    )
    if (
        tfidf_train_ids != train_counts.block_ids
        or tfidf_validation_ids != validation_counts.block_ids
    ):
        raise ValueError("Alinhamento TF-IDF diverge das contagens")
    checks["tfidf_alignment_matches_counts"] = True

    results_by_name["tfidf"] = evaluate_representation(
        "tfidf",
        tfidf_train,
        tfidf_validation,
        train_normal_positions,
        y_validation,
    )
    results = [results_by_name[name] for name in (
        "counts",
        "tfidf",
        "semantic",
        "semantic_plus_counts",
    )]

    report = {
        "stage": "etapa5_comparativo_representacoes_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
        },
        "protocol": {
            "partition_evaluated": "validation",
            "fit_partition": "train_normal",
            "detector": "OneClassSVM(kernel=rbf)",
            "nu": NU,
            "gamma": GAMMA,
            "scaler": "StandardScaler ajustado no train_normal completo",
            "subsample_size": SUBSAMPLE_SIZE,
            "seed": SEED,
            "threshold_selection": "max F1 (classe Anomaly)",
            "test_touched": False,
        },
        "validation": {
            "sessions": int(validation_counts_array.shape[0]),
            "anomalies": int(y_validation.sum()),
        },
        "inputs": {
            "counts_sha256": sha256_file(counts_path),
            "tfidf_manifest_sha256": sha256_file(tfidf_manifest_path),
            "semantic_manifest_sha256": sha256_file(semantic_manifest_path),
            "split_sha256": sha256_file(split_path),
        },
        "results": results,
        "acceptance": {
            "checks": checks,
            "accepted": all(checks.values()),
        },
    }
    detection.mkdir(parents=True, exist_ok=True)
    report_tmp = Path(f"{report_path}.tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(report_tmp, report_path)
    print(json.dumps({"accepted": True, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
