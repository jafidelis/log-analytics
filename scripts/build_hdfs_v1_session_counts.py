"""Etapa 4, incremento 4: vetores de contagem de templates por sessão.

Fonte: artefato de sequências do incremento 3 (validado por checksum).
Uma linha por sessão; colunas t1..t19 + unk; soma da linha == Length.
"""

import csv
import gzip
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from log_analytics.data.audit_hdfs_v1 import sha256_file

SEQUENCES_PATH = Path(
    "artifacts/representation/hdfs_v1_session_sequences.csv.gz"
)
SEQUENCES_MANIFEST_PATH = Path(
    "artifacts/representation/hdfs_v1_sequences_manifest.json"
)
TEMPLATES_PATH = Path("artifacts/parsing/hdfs_v1_templates.csv")
PARSING_MANIFEST_PATH = Path(
    "artifacts/parsing/hdfs_v1_parsing_manifest.json"
)

OUTPUT_DIR = Path("artifacts/representation")
COUNTS_PATH = OUTPUT_DIR / "hdfs_v1_session_counts.csv.gz"
MANIFEST_PATH = OUTPUT_DIR / "hdfs_v1_counts_manifest.json"

UNK_ID = 0


def load_accepted_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    acceptance = manifest["acceptance"]
    if acceptance.get("accepted") is not True or any(
        value is not True for value in acceptance["checks"].values()
    ):
        raise ValueError(f"Manifesto não aprovado: {path}")
    return manifest


def main() -> None:
    if COUNTS_PATH.exists() or MANIFEST_PATH.exists():
        raise SystemExit(
            "Artefatos já existem; remova-os deliberadamente para reexecutar."
        )

    # 1. Validar entradas.
    sequences_manifest = load_accepted_manifest(SEQUENCES_MANIFEST_PATH)
    parsing_manifest = load_accepted_manifest(PARSING_MANIFEST_PATH)

    sequences_sha256 = sha256_file(SEQUENCES_PATH)
    checks = {
        "sequences_sha256_matches_stage4": (
            sequences_sha256
            == sequences_manifest["outputs"]["session_sequences_csv_gz"][
                "sha256"
            ]
        ),
        "templates_sha256_matches_stage3": (
            sha256_file(TEMPLATES_PATH)
            == parsing_manifest["outputs"]["templates_csv"]["sha256"]
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    max_template_id = parsing_manifest["fit"]["clusters"]
    expected_sessions = sequences_manifest["results"]["sessions_by_split"]
    expected_unk = sequences_manifest["results"]["unk_by_split"]

    # Totais esperados por template/partição, vindos do catálogo da Etapa 3.
    expected_matches: dict[str, dict[int, int]] = {
        "train": {}, "validation": {}, "test": {},
    }
    with TEMPLATES_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            template_id = int(row["TemplateId"])
            expected_matches["train"][template_id] = int(row["MatchTrain"])
            expected_matches["validation"][template_id] = int(
                row["MatchValidation"]
            )
            expected_matches["test"][template_id] = int(row["MatchTest"])

    # 2. Processar em fluxo: entrada e saída ordenadas por BlockId.
    header = ["BlockId", "Split", "Length"] + [
        f"t{i}" for i in range(1, max_template_id + 1)
    ] + ["unk"]

    sessions_by_split: Counter = Counter()
    unk_by_split: Counter = Counter()
    totals: dict[str, Counter] = {
        "train": Counter(), "validation": Counter(), "test": Counter(),
    }
    all_rows_sum_to_length = True
    previous_block_id = ""
    sorted_and_unique = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts_tmp = Path(str(COUNTS_PATH) + ".tmp")
    with counts_tmp.open("wb") as raw_stream:
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", mtime=0
        ) as gz_stream:
            with io.TextIOWrapper(
                gz_stream, encoding="utf-8", newline=""
            ) as text_stream:
                writer = csv.writer(text_stream)
                writer.writerow(header)

                with gzip.open(
                    SEQUENCES_PATH, "rt", encoding="utf-8", newline=""
                ) as stream:
                    reader = csv.reader(stream)
                    if next(reader) != [
                        "BlockId", "Split", "Length", "Sequence"
                    ]:
                        raise ValueError("Cabeçalho inesperado.")

                    for block_id, split, length, sequence in reader:
                        if block_id <= previous_block_id:
                            sorted_and_unique = False
                        previous_block_id = block_id

                        ids = [int(i) for i in sequence.split(" ")]
                        counts = Counter(ids)

                        if sum(counts.values()) != int(length):
                            all_rows_sum_to_length = False

                        sessions_by_split[split] += 1
                        unk_by_split[split] += counts[UNK_ID]
                        totals[split].update(counts)

                        writer.writerow(
                            [block_id, split, length]
                            + [
                                counts[i]
                                for i in range(1, max_template_id + 1)
                            ]
                            + [counts[UNK_ID]]
                        )
    os.replace(counts_tmp, COUNTS_PATH)

    # 3. Checks de conservação.
    checks["input_sorted_and_unique"] = sorted_and_unique
    checks["all_rows_sum_to_length"] = all_rows_sum_to_length
    checks["sessions_by_split_match_sequences"] = (
        dict(sessions_by_split) == expected_sessions
    )
    checks["unk_by_split_match_sequences"] = (
        dict(unk_by_split) == expected_unk
    )
    checks["template_totals_match_stage3_catalog"] = all(
        totals[split][template_id]
        == expected_matches[split][template_id]
        for split in ("train", "validation", "test")
        for template_id in range(1, max_template_id + 1)
    )

    # 4. Manifesto.
    accepted = all(checks.values())
    manifest = {
        "stage": "etapa4_incremento4_contagens_por_sessao",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0]},
        "representation": {
            "unit": "session (BlockId)",
            "columns": header,
            "vocabulary": f"t1..t{max_template_id} + unk (UNK={UNK_ID})",
            "label_included": False,
            "derived_from": "hdfs_v1_session_sequences.csv.gz",
        },
        "inputs": {
            "sequences_sha256": sequences_sha256,
            "templates_sha256": sha256_file(TEMPLATES_PATH),
            "sequences_manifest_sha256": sha256_file(
                SEQUENCES_MANIFEST_PATH
            ),
        },
        "results": {
            "sessions_by_split": dict(sessions_by_split),
            "unk_by_split": dict(unk_by_split),
        },
        "outputs": {
            "session_counts_csv_gz": {
                "path": str(COUNTS_PATH),
                "sha256": sha256_file(COUNTS_PATH),
                "size_bytes": COUNTS_PATH.stat().st_size,
            },
        },
        "acceptance": {"checks": checks, "accepted": accepted},
    }
    manifest_tmp = Path(str(MANIFEST_PATH) + ".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, MANIFEST_PATH)

    print(json.dumps(
        {"accepted": accepted, "checks": checks,
         "results": manifest["results"]},
        ensure_ascii=False, indent=2,
    ))

    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()