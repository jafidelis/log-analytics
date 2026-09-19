"""Etapa 4B.3: ajuste e transformação TF-IDF por sessão."""

import io
import csv
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import save_npz, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.text_features_hdfs_v1 import (
    analyze_log_document,
)

CORPUS_PATH = Path(
    "artifacts/representation/hdfs_v1_session_corpus.csv.gz"
)
CORPUS_MANIFEST_PATH = Path(
    "artifacts/representation/hdfs_v1_session_corpus_manifest.json"
)
SPLIT_PATH = Path(
    "artifacts/data_splits/hdfs_v1_session_split.csv"
)
SPLIT_MANIFEST_PATH = Path(
    "artifacts/data_splits/hdfs_v1_split_manifest.json"
)
TRAIN_NORMAL_PATH = Path(
    "artifacts/data_splits/hdfs_v1_train_normal_block_ids.txt"
)

OUTPUT_DIR = Path("artifacts/representation")
VECTORIZER_PATH = OUTPUT_DIR / "hdfs_v1_tfidf_vectorizer.joblib"
INDEX_PATH = OUTPUT_DIR / "hdfs_v1_tfidf_session_index.csv.gz"
MANIFEST_PATH = OUTPUT_DIR / "hdfs_v1_tfidf_manifest.json"

MATRIX_PATHS = {
    "train": OUTPUT_DIR / "hdfs_v1_tfidf_train.npz",
    "validation": OUTPUT_DIR / "hdfs_v1_tfidf_validation.npz",
    "test": OUTPUT_DIR / "hdfs_v1_tfidf_test.npz",
}

CORPUS_HEADER = [
    "BlockId",
    "Split",
    "Length",
    "Document",
    "DocumentHash",
]

BATCH_SIZE = 4096

def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    acceptance = manifest["acceptance"]
    checks = acceptance.get("checks")

    if (
        acceptance.get("accepted") is not True
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError(f"Manifesto não aprovado: {path}")

    return manifest


def load_train_normal_ids() -> set[str]:
    return {
        line.strip()
        for line in TRAIN_NORMAL_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }


def read_corpus_rows():
    with gzip.open(
        CORPUS_PATH,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.reader(stream)

        if next(reader, None) != CORPUS_HEADER:
            raise ValueError("Cabeçalho inválido no corpus.")

        for row in reader:
            if len(row) != 5:
                raise ValueError("Linha inválida no corpus.")

            block_id, split, length, document, document_hash = row

            yield (
                block_id,
                split,
                int(length),
                document,
                document_hash,
            )


def iter_train_normal_documents(train_normal_ids: set[str]):
    for block_id, split, _, document, _ in read_corpus_rows():
        if split == "train" and block_id in train_normal_ids:
            yield document


def count_train_normal_documents(
    train_normal_ids: set[str],
) -> int:
    return sum(
        1
        for _ in iter_train_normal_documents(train_normal_ids)
    )


def transform_split(
    vectorizer: TfidfVectorizer,
    split: str,
):
    matrices = []
    index_rows = []
    row_number = 0

    documents = []

    for block_id, row_split, _, document, _ in read_corpus_rows():
        if row_split != split:
            continue

        documents.append(document)
        index_rows.append((row_number, block_id, split))
        row_number += 1

        if len(documents) == BATCH_SIZE:
            matrices.append(vectorizer.transform(documents))
            documents = []

    if documents:
        matrices.append(vectorizer.transform(documents))

    if not matrices:
        raise ValueError(f"Nenhum documento encontrado: {split}")

    return vstack(matrices, format="csr"), index_rows

def main() -> None:
    outputs = [
        VECTORIZER_PATH,
        INDEX_PATH,
        MANIFEST_PATH,
        *MATRIX_PATHS.values(),
    ]

    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    corpus_manifest = load_manifest(CORPUS_MANIFEST_PATH)
    split_manifest = load_manifest(SPLIT_MANIFEST_PATH)

    checks = {
        "corpus_sha256_matches_manifest": (
            sha256_file(CORPUS_PATH)
            == corpus_manifest["outputs"][
                "session_corpus_csv_gz"
            ]["sha256"]
        ),
        "split_sha256_matches_manifest": (
            sha256_file(SPLIT_PATH)
            == split_manifest["outputs"][
                "master_csv"
            ]["sha256"]
        ),
    }

    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    train_normal_ids = load_train_normal_ids()
    fit_documents = count_train_normal_documents(
        train_normal_ids
    )

    expected_fit_documents = 390_756
    checks["fit_documents_expected"] = (
        fit_documents == expected_fit_documents
    )

    if not checks["fit_documents_expected"]:
        raise SystemExit(f"Quantidade inválida: {checks}")

    vectorizer = TfidfVectorizer(
        analyzer=analyze_log_document,
        dtype=np.float32,
        norm="l2",
        sublinear_tf=True,
    )

    vectorizer.fit(
        iter_train_normal_documents(train_normal_ids)
    )

    split_matrices = {}
    all_index_rows = []
    expected_sessions = {
        "train": 402_542,
        "validation": 86_260,
        "test": 86_259,
    }

    for split in ("train", "validation", "test"):
        matrix, index_rows = transform_split(
            vectorizer,
            split,
        )

        split_matrices[split] = matrix
        all_index_rows.extend(index_rows)

        checks[f"{split}_rows_expected"] = (
            matrix.shape[0] == expected_sessions[split]
        )
        checks[f"{split}_dimensions_stable"] = (
            matrix.shape[1] == len(vectorizer.vocabulary_)
        )

    checks["feature_count_positive"] = (
        len(vectorizer.vocabulary_) > 0
    )

    if not all(checks.values()):
        raise SystemExit(f"Critérios reprovados: {checks}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, matrix in split_matrices.items():
        save_npz(MATRIX_PATHS[split], matrix)

    joblib.dump(vectorizer, VECTORIZER_PATH)

    with INDEX_PATH.open("wb") as raw_output:
      with gzip.GzipFile(
          filename="",
          fileobj=raw_output,
          mode="wb",
          mtime=0,
      ) as gzip_output:
          with io.TextIOWrapper(
              gzip_output,
              encoding="utf-8",
              newline="",
          ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["Row", "BlockId", "Split"])
            writer.writerows(all_index_rows)

    manifest = {
        "stage": "etapa4b_incremento3_tfidf",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "representation": {
            "unit": "sessão",
            "analyzer": "analyze_log_document",
            "ngram_range": [1, 2],
            "fit_scope": "train_normal",
            "labels_in_features": False,
            "dtype": "float32",
            "normalization": "l2",
            "sublinear_tf": True,
        },
        "inputs": {
            "corpus_sha256": sha256_file(CORPUS_PATH),
            "split_sha256": sha256_file(SPLIT_PATH),
            "corpus_manifest_sha256": sha256_file(
                CORPUS_MANIFEST_PATH
            ),
        },
        "fit": {
            "documents": fit_documents,
            "features": len(vectorizer.vocabulary_),
        },
        "outputs": {
            "vectorizer": {
                "path": str(VECTORIZER_PATH),
                "sha256": sha256_file(VECTORIZER_PATH),
            },
            "session_index": {
                "path": str(INDEX_PATH),
                "sha256": sha256_file(INDEX_PATH),
            },
            "matrices": {
                split: {
                    "path": str(MATRIX_PATHS[split]),
                    "rows": split_matrices[split].shape[0],
                    "columns": split_matrices[split].shape[1],
                    "sha256": sha256_file(MATRIX_PATHS[split]),
                }
                for split in split_matrices
            },
        },
        "acceptance": {
            "checks": checks,
            "accepted": all(checks.values()),
        },
    }

    manifest_tmp = Path(f"{MANIFEST_PATH}.tmp")
    manifest_tmp.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, MANIFEST_PATH)

    print(
        json.dumps(
            manifest["acceptance"],
            ensure_ascii=False,
            indent=2,
        )
    )

if __name__ == "__main__":
    main()