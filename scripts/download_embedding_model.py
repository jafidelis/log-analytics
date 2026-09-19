"""Baixa e registra o modelo de embeddings."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import sentence_transformers

MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "models" / "huggingface"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "embedding_model_manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(CACHE_DIR),
        )
    )

    model = SentenceTransformer(str(snapshot_path))

    texts = [
        "Received block <BLOCK_ID> from <IP>",
        "Bloco recebido <BLOCK_ID> de <IP>",
    ]

    first = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    second = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    relative_snapshot = snapshot_path.relative_to(
        PROJECT_ROOT
    )

    files = {}
    for path in sorted(snapshot_path.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(PROJECT_ROOT))] = (
                sha256_file(path)
            )

    manifest = {
        "model_id": MODEL_ID,
        "snapshot_path": str(relative_snapshot),
        "snapshot_revision": snapshot_path.name,
        "license": "apache-2.0",
        "library": {
            "name": "sentence-transformers",
            "version": sentence_transformers.__version__,
        },
        "model": {
            "embedding_dimension": (
                model.get_sentence_embedding_dimension()
            ),
            "max_seq_length": model.max_seq_length,
            "normalize_embeddings": True,
        },
        "smoke_test": {
            "texts": len(texts),
            "shape": list(first.shape),
            "finite": bool(np.isfinite(first).all()),
            "max_abs_difference": float(
                np.max(np.abs(first - second))
            ),
        },
        "files_sha256": files,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if not np.isfinite(first).all():
        raise ValueError("Embedding não finito.")

    if first.shape != (2, 384):
        raise ValueError(
            f"Dimensão inesperada: {first.shape}"
        )

    if not np.allclose(first, second, atol=1e-6):
        raise ValueError(
            "O modelo não foi determinístico no teste."
        )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "snapshot_path": str(relative_snapshot),
                "dimension": first.shape[1],
                "max_seq_length": model.max_seq_length,
                "deterministic": True,
                "manifest": str(
                    MANIFEST_PATH.relative_to(PROJECT_ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()