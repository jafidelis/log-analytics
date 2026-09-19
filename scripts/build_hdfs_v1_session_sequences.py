"""Etapa 4, incremento 3: sequências de templates por sessão.

Fonte única: parsed_events da Etapa 3 (validado por checksum).
Saída: uma linha por sessão, ordenada lexicalmente por BlockId.
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
from log_analytics.data.session_hdfs_v1 import load_split_index

PARSED_EVENTS_PATH = Path("artifacts/parsing/hdfs_v1_parsed_events.csv.gz")
PARSING_MANIFEST_PATH = Path("artifacts/parsing/hdfs_v1_parsing_manifest.json")
SPLIT_CSV_PATH = Path("artifacts/data_splits/hdfs_v1_session_split.csv")
SPLIT_MANIFEST_PATH = Path(
    "artifacts/data_splits/hdfs_v1_split_manifest.json"
)

OUTPUT_DIR = Path("artifacts/representation")
SEQUENCES_PATH = OUTPUT_DIR / "hdfs_v1_session_sequences.csv.gz"
MANIFEST_PATH = OUTPUT_DIR / "hdfs_v1_sequences_manifest.json"

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
    if SEQUENCES_PATH.exists() or MANIFEST_PATH.exists():
        raise SystemExit(
            "Artefatos já existem; remova-os deliberadamente para reexecutar."
        )

    # 1. Validar entradas contra os manifestos aprovados.
    parsing_manifest = load_accepted_manifest(PARSING_MANIFEST_PATH)
    split_manifest = load_accepted_manifest(SPLIT_MANIFEST_PATH)

    parsed_sha256 = sha256_file(PARSED_EVENTS_PATH)
    split_sha256 = sha256_file(SPLIT_CSV_PATH)
    checks = {
        "parsed_events_sha256_matches_stage3": (
            parsed_sha256
            == parsing_manifest["outputs"]["parsed_events_csv_gz"]["sha256"]
        ),
        "split_csv_sha256_matches_stage2": (
            split_sha256
            == split_manifest["outputs"]["master_csv"]["sha256"]
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    # Valores esperados vêm dos manifestos, não de constantes locais.
    expected_events = parsing_manifest["transform"]["events_by_split"]
    expected_unk = parsing_manifest["transform"]["unk_by_split"]
    expected_sessions = parsing_manifest["transform"]["sessions_by_split"]
    max_template_id = parsing_manifest["fit"]["clusters"]

    split_index = load_split_index(SPLIT_CSV_PATH)

    # 2. Agrupar os eventos por sessão, na ordem do arquivo (= LineId).
    sequences: dict[str, list[int]] = {}
    unk_by_split: Counter = Counter()
    with gzip.open(
        PARSED_EVENTS_PATH, "rt", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["LineId", "BlockId", "Split", "TemplateId"]:
            raise ValueError(f"Cabeçalho inesperado: {header}")

        for _line_id, block_id, split, template_id in reader:
            if template_id == "UNK_TEMPLATE":
                numeric_id = UNK_ID
                unk_by_split[split] += 1
            else:
                numeric_id = int(template_id)
                if not 1 <= numeric_id <= max_template_id:
                    raise ValueError(
                        f"TemplateId fora do vocabulário: {template_id}"
                    )
            sequences.setdefault(block_id, []).append(numeric_id)

    # 3. Consolidar contagens por partição.
    sessions_by_split: Counter = Counter()
    events_by_split: Counter = Counter()
    for block_id, sequence in sequences.items():
        split = split_index[block_id].split
        sessions_by_split[split] += 1
        events_by_split[split] += len(sequence)

    checks["all_sessions_present"] = (
        len(sequences) == len(split_index)
    )
    checks["no_empty_sequence"] = all(
        len(sequence) > 0 for sequence in sequences.values()
    )
    checks["sessions_by_split_match_stage3"] = (
        dict(sessions_by_split) == expected_sessions
    )
    checks["events_by_split_match_stage3"] = (
        dict(events_by_split) == expected_events
    )
    checks["unk_by_split_match_stage3"] = (
        dict(unk_by_split) == expected_unk
    )

    # 4. Escrever ordenado por BlockId, atômico e determinístico.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sequences_tmp = Path(str(SEQUENCES_PATH) + ".tmp")
    with sequences_tmp.open("wb") as raw_stream:
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", mtime=0
        ) as gz_stream:
            with io.TextIOWrapper(
                gz_stream, encoding="utf-8", newline=""
            ) as text_stream:
                writer = csv.writer(text_stream)
                writer.writerow(
                    ["BlockId", "Split", "Length", "Sequence"]
                )
                for block_id in sorted(sequences):
                    sequence = sequences[block_id]
                    writer.writerow([
                        block_id,
                        split_index[block_id].split,
                        len(sequence),
                        " ".join(str(i) for i in sequence),
                    ])
    os.replace(sequences_tmp, SEQUENCES_PATH)

    # 5. Manifesto.
    accepted = all(checks.values())
    manifest = {
        "stage": "etapa4_incremento3_sequencias_por_sessao",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0]},
        "representation": {
            "unit": "session (BlockId)",
            "order": "LineId ascendente dentro da sessão",
            "vocabulary": f"1..{max_template_id} + UNK={UNK_ID}",
            "label_included": False,
        },
        "inputs": {
            "parsed_events_sha256": parsed_sha256,
            "split_csv_sha256": split_sha256,
            "parsing_manifest_sha256": sha256_file(PARSING_MANIFEST_PATH),
        },
        "results": {
            "sessions_by_split": dict(sessions_by_split),
            "events_by_split": dict(events_by_split),
            "unk_by_split": dict(unk_by_split),
        },
        "outputs": {
            "session_sequences_csv_gz": {
                "path": str(SEQUENCES_PATH),
                "sha256": sha256_file(SEQUENCES_PATH),
                "size_bytes": SEQUENCES_PATH.stat().st_size,
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