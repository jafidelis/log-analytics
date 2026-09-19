"""Audita o catálogo textual e o tokenizer do encoder semântico HDFS v1.

Este incremento não gera embeddings e não lê rótulos. Ele valida as entradas,
constrói o catálogo dos textos de evento e verifica se o tokenizer congelado
truncaria ou substituiria algum token por seu token desconhecido nativo.
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
from typing import Any, Iterable

from sentence_transformers import SentenceTransformer

from log_analytics.data.audit_hdfs_v1 import sha256_file


MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
EXPECTED_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EXPECTED_TEMPLATE_COUNT = 19
EXPECTED_UNK_EVENTS = 5_484
EXPECTED_UNIQUE_UNK = 266
EXPECTED_UNIQUE_TEXTS = 285
EXPECTED_TOTAL_EVENTS = 11_175_629
EXPECTED_EMBEDDING_DIMENSION = 384
EXPECTED_MAX_SEQ_LENGTH = 128

TEMPLATE_HEADER = [
    "TemplateId",
    "FitCount",
    "MatchTrain",
    "MatchValidation",
    "MatchTest",
    "MatchNone",
    "Template",
]
UNKNOWN_HEADER = [
    "LineId",
    "BlockId",
    "Split",
    "PreparedText",
    "TextHash",
]
CATALOG_HEADER = [
    "CatalogIndex",
    "SourceType",
    "SourceId",
    "Text",
    "TextHash",
    "Occurrences",
]
AUDIT_HEADER = [
    "CatalogIndex",
    "TextHash",
    "TokenCountTotal",
    "TokenCountContent",
    "SpecialTokenCount",
    "TokenizerUnkCount",
    "MaxSeqLength",
    "WouldBeTruncated",
    "TruncatedTokenCount",
    "TokenIdsJson",
    "TokensJson",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Constrói o catálogo semântico do HDFS v1 e audita o "
            "tokenizer sem gerar embeddings."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Raiz do projeto log-analytics.",
    )
    return parser.parse_args()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON deve conter um objeto: {path}")
    return value


def load_accepted_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    acceptance = manifest.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"Manifesto sem bloco acceptance: {path}")

    checks = acceptance.get("checks")
    if (
        acceptance.get("accepted") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError(f"Manifesto não aprovado: {path}")

    return manifest


def resolve_under_root(
    project_root: Path,
    relative_path: str,
) -> Path:
    if not relative_path:
        raise ValueError("Caminho relativo vazio.")

    root = project_root.resolve(strict=True)
    candidate = (root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Caminho fora da raiz do projeto: {relative_path}"
        ) from error
    return candidate


def verify_model_files(
    project_root: Path,
    expected_files: dict[str, str],
) -> bool:
    if not expected_files:
        return False

    for relative_path, expected_sha256 in expected_files.items():
        path = resolve_under_root(project_root, relative_path)
        if not path.is_file() or sha256_file(path) != expected_sha256:
            return False
    return True


def load_template_entries(
    path: Path,
) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    template_ids: set[int] = set()
    total_occurrences = 0

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != TEMPLATE_HEADER:
            raise ValueError(f"Cabeçalho inesperado: {path}")

        for row in reader:
            template_id = int(row["TemplateId"])
            if template_id in template_ids:
                raise ValueError(f"TemplateId duplicado: {template_id}")
            template_ids.add(template_id)

            text = row["Template"]
            if not text:
                raise ValueError(f"Template vazio: {template_id}")

            occurrences = sum(
                int(row[column])
                for column in (
                    "MatchTrain",
                    "MatchValidation",
                    "MatchTest",
                    "MatchNone",
                )
            )
            if occurrences <= 0:
                raise ValueError(
                    f"Template sem ocorrência: {template_id}"
                )

            entries.append(
                {
                    "SourceType": "template",
                    "SourceId": str(template_id),
                    "Text": text,
                    "TextHash": hash_text(text),
                    "Occurrences": occurrences,
                    "sort_key": (0, template_id),
                }
            )
            total_occurrences += occurrences

    expected_ids = set(range(1, EXPECTED_TEMPLATE_COUNT + 1))
    if template_ids != expected_ids:
        raise ValueError(
            "Catálogo de templates não corresponde aos ids 1..19."
        )

    return entries, total_occurrences


def load_unknown_entries(
    path: Path,
) -> tuple[list[dict[str, Any]], int]:
    text_by_hash: dict[str, str] = {}
    occurrences: Counter[str] = Counter()
    previous_line_id = 0
    total_events = 0

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != UNKNOWN_HEADER:
            raise ValueError(f"Cabeçalho inesperado: {path}")

        for row in reader:
            line_id = int(row["LineId"])
            if line_id <= previous_line_id:
                raise ValueError(
                    "LineId de UNK duplicado ou fora de ordem."
                )
            previous_line_id = line_id

            text = row["PreparedText"]
            text_hash = row["TextHash"]
            if not text:
                raise ValueError(f"Texto UNK vazio na linha {line_id}.")
            if hash_text(text) != text_hash:
                raise ValueError(
                    f"TextHash inválido na linha {line_id}."
                )

            known_text = text_by_hash.setdefault(text_hash, text)
            if known_text != text:
                raise ValueError(
                    f"Colisão SHA-256 detectada: {text_hash}"
                )

            occurrences[text_hash] += 1
            total_events += 1

    entries = [
        {
            "SourceType": "unk_prepared",
            "SourceId": text_hash,
            "Text": text_by_hash[text_hash],
            "TextHash": text_hash,
            "Occurrences": occurrences[text_hash],
            "sort_key": (1, text_hash),
        }
        for text_hash in sorted(text_by_hash)
    ]
    return entries, total_events


def build_catalog(
    templates_path: Path,
    unknown_events_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    template_entries, template_occurrences = load_template_entries(
        templates_path
    )
    unknown_entries, unknown_events = load_unknown_entries(
        unknown_events_path
    )

    template_hashes = {
        entry["TextHash"] for entry in template_entries
    }
    unknown_hashes = {entry["TextHash"] for entry in unknown_entries}
    overlap = template_hashes & unknown_hashes
    if overlap:
        raise ValueError(
            "Há textos compartilhados entre templates e UNK: "
            f"{sorted(overlap)}"
        )

    combined_entries = [*template_entries, *unknown_entries]
    text_by_hash: dict[str, str] = {}
    seen_texts: set[str] = set()
    for entry in combined_entries:
        text_hash = entry["TextHash"]
        text = entry["Text"]

        known_text = text_by_hash.setdefault(text_hash, text)
        if known_text != text:
            raise ValueError(
                f"Colisão SHA-256 global detectada: {text_hash}"
            )
        if text in seen_texts:
            raise ValueError(f"Texto duplicado no catálogo: {text}")
        seen_texts.add(text)

    catalog = sorted(
        combined_entries,
        key=lambda entry: entry["sort_key"],
    )
    for index, entry in enumerate(catalog):
        entry["CatalogIndex"] = index
        entry.pop("sort_key")

    summary = {
        "templates": len(template_entries),
        "template_occurrences": template_occurrences,
        "unknown_events": unknown_events,
        "unique_unknown_texts": len(unknown_entries),
        "unique_texts": len(seen_texts),
        "total_occurrences": template_occurrences + unknown_events,
        "hash_collisions": 0,
        "duplicate_texts": 0,
        "hash_overlap": len(overlap),
    }
    return catalog, summary


def audit_tokenizer(
    catalog: list[dict[str, Any]],
    model: SentenceTransformer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = model.tokenizer
    max_seq_length = int(model.max_seq_length)
    unknown_token_id = tokenizer.unk_token_id

    audit_rows: list[dict[str, Any]] = []
    total_counts: list[int] = []
    truncated_texts = 0
    tokenizer_unknown_tokens = 0
    deterministic = True
    ids_tokens_aligned = True
    special_masks_valid = True

    for entry in catalog:
        encoded = tokenizer(
            entry["Text"],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_special_tokens_mask=True,
        )
        repeated = tokenizer(
            entry["Text"],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_special_tokens_mask=True,
        )

        token_ids = list(encoded["input_ids"])
        special_mask = list(encoded["special_tokens_mask"])
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        deterministic = deterministic and (
            token_ids == list(repeated["input_ids"])
            and special_mask
            == list(repeated["special_tokens_mask"])
        )
        ids_tokens_aligned = ids_tokens_aligned and (
            len(token_ids) == len(tokens)
        )
        special_masks_valid = special_masks_valid and (
            len(special_mask) == len(token_ids)
            and all(value in (0, 1) for value in special_mask)
        )

        total_count = len(token_ids)
        special_count = sum(special_mask)
        content_count = total_count - special_count
        unknown_count = (
            token_ids.count(unknown_token_id)
            if unknown_token_id is not None
            else 0
        )
        would_be_truncated = total_count > max_seq_length
        truncated_count = max(0, total_count - max_seq_length)

        truncated_texts += int(would_be_truncated)
        tokenizer_unknown_tokens += unknown_count
        total_counts.append(total_count)

        audit_rows.append(
            {
                "CatalogIndex": entry["CatalogIndex"],
                "TextHash": entry["TextHash"],
                "TokenCountTotal": total_count,
                "TokenCountContent": content_count,
                "SpecialTokenCount": special_count,
                "TokenizerUnkCount": unknown_count,
                "MaxSeqLength": max_seq_length,
                "WouldBeTruncated": str(
                    would_be_truncated
                ).lower(),
                "TruncatedTokenCount": truncated_count,
                "TokenIdsJson": json.dumps(
                    token_ids,
                    separators=(",", ":"),
                ),
                "TokensJson": json.dumps(
                    tokens,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

    sorted_counts = sorted(total_counts)
    p95_index = max(0, (95 * len(sorted_counts) + 99) // 100 - 1)
    summary = {
        "texts": len(audit_rows),
        "min_token_count": min(sorted_counts),
        "p95_token_count": sorted_counts[p95_index],
        "max_token_count": max(sorted_counts),
        "truncated_texts": truncated_texts,
        "tokenizer_unknown_tokens": tokenizer_unknown_tokens,
        "deterministic": deterministic,
        "ids_tokens_aligned": ids_tokens_aligned,
        "special_masks_valid": special_masks_valid,
    }
    return audit_rows, summary


def write_gzip_csv(
    path: Path,
    header: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("wb") as raw_output:
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
            ) as text_output:
                writer = csv.DictWriter(
                    text_output,
                    fieldnames=header,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve(strict=True)

    templates_path = (
        project_root / "artifacts/parsing/hdfs_v1_templates.csv"
    )
    parsing_manifest_path = (
        project_root
        / "artifacts/parsing/hdfs_v1_parsing_manifest.json"
    )
    unknown_events_path = (
        project_root
        / "artifacts/representation/hdfs_v1_unknown_events.csv.gz"
    )
    unknown_manifest_path = (
        project_root
        / "artifacts/representation/hdfs_v1_unknown_events_manifest.json"
    )
    model_manifest_path = (
        project_root
        / "artifacts/models/embedding_model_manifest.json"
    )

    output_dir = project_root / "artifacts/representation"
    catalog_path = (
        output_dir / "hdfs_v1_semantic_event_catalog.csv.gz"
    )
    audit_path = (
        output_dir / "hdfs_v1_tokenization_audit.csv.gz"
    )
    manifest_path = (
        output_dir / "hdfs_v1_semantic_gate_manifest.json"
    )

    outputs = [catalog_path, audit_path, manifest_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    parsing_manifest = load_accepted_manifest(parsing_manifest_path)
    unknown_manifest = load_accepted_manifest(unknown_manifest_path)
    model_manifest = load_json(model_manifest_path)

    snapshot_path = resolve_under_root(
        project_root,
        model_manifest["snapshot_path"],
    )

    upstream_checks = {
        "templates_sha256_matches_parsing": (
            sha256_file(templates_path)
            == parsing_manifest["outputs"]["templates_csv"]["sha256"]
        ),
        "unknown_sha256_matches_manifest": (
            sha256_file(unknown_events_path)
            == unknown_manifest["outputs"]["unknown_events_csv_gz"][
                "sha256"
            ]
        ),
        "unknown_links_to_parsing_manifest": (
            sha256_file(parsing_manifest_path)
            == unknown_manifest["inputs"]["parsing_manifest_sha256"]
        ),
        "unknown_links_to_parsed_events": (
            unknown_manifest["inputs"]["parsed_events_sha256"]
            == parsing_manifest["outputs"]["parsed_events_csv_gz"][
                "sha256"
            ]
        ),
        "model_id_expected": (
            model_manifest.get("model_id") == MODEL_ID
        ),
        "model_revision_expected": (
            model_manifest.get("snapshot_revision")
            == EXPECTED_MODEL_REVISION
            == snapshot_path.name
        ),
        "model_manifest_dimension_expected": (
            model_manifest["model"]["embedding_dimension"]
            == EXPECTED_EMBEDDING_DIMENSION
        ),
        "model_manifest_max_length_expected": (
            model_manifest["model"]["max_seq_length"]
            == EXPECTED_MAX_SEQ_LENGTH
        ),
        "sentence_transformers_version_matches_manifest": (
            version("sentence-transformers")
            == model_manifest["library"]["version"]
        ),
        "model_files_sha256_match_manifest": verify_model_files(
            project_root,
            model_manifest.get("files_sha256", {}),
        ),
    }
    if not all(upstream_checks.values()):
        raise SystemExit(f"Entradas inválidas: {upstream_checks}")

    catalog, catalog_summary = build_catalog(
        templates_path,
        unknown_events_path,
    )

    model = SentenceTransformer(
        str(snapshot_path),
        local_files_only=True,
        device="cpu",
    )
    model.eval()

    audit_rows, tokenizer_summary = audit_tokenizer(catalog, model)

    checks = {
        **upstream_checks,
        "template_count_expected": (
            catalog_summary["templates"] == EXPECTED_TEMPLATE_COUNT
        ),
        "unknown_event_count_expected": (
            catalog_summary["unknown_events"] == EXPECTED_UNK_EVENTS
        ),
        "unique_unknown_count_expected": (
            catalog_summary["unique_unknown_texts"]
            == EXPECTED_UNIQUE_UNK
        ),
        "unique_text_count_expected": (
            catalog_summary["unique_texts"]
            == EXPECTED_UNIQUE_TEXTS
        ),
        "total_occurrences_expected": (
            catalog_summary["total_occurrences"]
            == EXPECTED_TOTAL_EVENTS
        ),
        "no_template_unknown_hash_overlap": (
            catalog_summary["hash_overlap"] == 0
        ),
        "no_hash_collisions": (
            catalog_summary["hash_collisions"] == 0
        ),
        "no_duplicate_texts": (
            catalog_summary["duplicate_texts"] == 0
        ),
        "model_dimension_expected": (
            model.get_embedding_dimension()
            == EXPECTED_EMBEDDING_DIMENSION
        ),
        "model_max_length_expected": (
            int(model.max_seq_length) == EXPECTED_MAX_SEQ_LENGTH
        ),
        "tokenizer_audited_all_texts": (
            tokenizer_summary["texts"] == EXPECTED_UNIQUE_TEXTS
        ),
        "tokenization_deterministic": tokenizer_summary[
            "deterministic"
        ],
        "token_ids_and_tokens_aligned": tokenizer_summary[
            "ids_tokens_aligned"
        ],
        "special_token_masks_valid": tokenizer_summary[
            "special_masks_valid"
        ],
        "no_text_would_be_truncated": (
            tokenizer_summary["truncated_texts"] == 0
        ),
        "no_tokenizer_unknown_tokens": (
            tokenizer_summary["tokenizer_unknown_tokens"] == 0
        ),
    }
    accepted = all(checks.values())
    if not accepted:
        raise SystemExit(
            json.dumps(
                {
                    "accepted": False,
                    "checks": checks,
                    "catalog": catalog_summary,
                    "tokenizer": tokenizer_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_tmp = Path(f"{catalog_path}.tmp")
    audit_tmp = Path(f"{audit_path}.tmp")
    manifest_tmp = Path(f"{manifest_path}.tmp")

    write_gzip_csv(catalog_tmp, CATALOG_HEADER, catalog)
    write_gzip_csv(audit_tmp, AUDIT_HEADER, audit_rows)

    os.replace(catalog_tmp, catalog_path)
    os.replace(audit_tmp, audit_path)

    manifest = {
        "stage": "etapa4b_gate_semantico_tokenizacao",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
            "torch": version("torch"),
        },
        "model": {
            "model_id": MODEL_ID,
            "snapshot_path": str(snapshot_path.relative_to(project_root)),
            "snapshot_revision": snapshot_path.name,
            "embedding_dimension": (
                model.get_embedding_dimension()
            ),
            "max_seq_length": int(model.max_seq_length),
            "device": "cpu",
            "embeddings_generated": False,
        },
        "tokenization": {
            "add_special_tokens": True,
            "truncation": False,
            "padding": False,
            "limit_source": "SentenceTransformer.max_seq_length",
            "tokenizer_unknown_token": model.tokenizer.unk_token,
            "tokenizer_unknown_token_id": model.tokenizer.unk_token_id,
        },
        "inputs": {
            "templates_sha256": sha256_file(templates_path),
            "parsing_manifest_sha256": sha256_file(
                parsing_manifest_path
            ),
            "unknown_events_sha256": sha256_file(
                unknown_events_path
            ),
            "unknown_manifest_sha256": sha256_file(
                unknown_manifest_path
            ),
            "embedding_model_manifest_sha256": sha256_file(
                model_manifest_path
            ),
        },
        "results": {
            "catalog": catalog_summary,
            "tokenizer": tokenizer_summary,
        },
        "outputs": {
            "semantic_event_catalog_csv_gz": {
                "path": str(catalog_path.relative_to(project_root)),
                "sha256": sha256_file(catalog_path),
                "size_bytes": catalog_path.stat().st_size,
            },
            "tokenization_audit_csv_gz": {
                "path": str(audit_path.relative_to(project_root)),
                "sha256": sha256_file(audit_path),
                "size_bytes": audit_path.stat().st_size,
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
    os.replace(manifest_tmp, manifest_path)

    print(
        json.dumps(
            {
                "accepted": accepted,
                "catalog": catalog_summary,
                "tokenizer": tokenizer_summary,
                "manifest": str(manifest_path.relative_to(project_root)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
