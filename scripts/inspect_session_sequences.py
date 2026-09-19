"""Etapa 4, incremento 1: reconstruir e conferir sequências de 3 sessões."""

import csv
import gzip
import json
from pathlib import Path

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.session_hdfs_v1 import load_split_index

PARSED_EVENTS_PATH = Path("artifacts/parsing/hdfs_v1_parsed_events.csv.gz")
PARSING_MANIFEST_PATH = Path("artifacts/parsing/hdfs_v1_parsing_manifest.json")
TEMPLATES_PATH = Path("artifacts/parsing/hdfs_v1_templates.csv")
SPLIT_CSV_PATH = Path("artifacts/data_splits/hdfs_v1_session_split.csv")
RAW_LOG_PATH = Path("data/raw/hdfs_v1/HDFS.log")

UNK_ID = 0
SHOW_EVENTS = 8
SHOW_SEQUENCE = 30


def main() -> None:
    # 1. Validar a fronteira: só trabalhamos sobre artefato aceito.
    manifest = json.loads(
        PARSING_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if manifest["acceptance"]["accepted"] is not True:
        raise ValueError("Manifesto da etapa 3 não aprovado.")

    expected_sha256 = manifest["outputs"]["parsed_events_csv_gz"]["sha256"]
    if sha256_file(PARSED_EVENTS_PATH) != expected_sha256:
        raise ValueError("parsed_events diverge do manifesto da etapa 3.")

    split_index = load_split_index(SPLIT_CSV_PATH)

    # 2. Escolher 3 sessões deterministicamente.
    known_train = "blk_-1608999687919862906"
    first_anomaly = min(
        block_id
        for block_id, assignment in split_index.items()
        if assignment.label == "Anomaly"
    )
    first_validation = min(
        block_id
        for block_id, assignment in split_index.items()
        if assignment.split == "validation"
    )
    targets = [known_train, first_anomaly, first_validation]

    # 3. Carregar o catálogo de templates.
    templates: dict[int, str] = {UNK_ID: "UNK_TEMPLATE"}
    with TEMPLATES_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            templates[int(row["TemplateId"])] = row["Template"]

    # 4. Reconstruir as sequências das sessões-alvo.
    sequences: dict[str, list[tuple[int, int]]] = {
        block_id: [] for block_id in targets
    }
    with gzip.open(
        PARSED_EVENTS_PATH, "rt", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["LineId", "BlockId", "Split", "TemplateId"]:
            raise ValueError(f"Cabeçalho inesperado: {header}")

        for line_id, block_id, _split, template_id in reader:
            if block_id in sequences:
                numeric_id = (
                    UNK_ID
                    if template_id == "UNK_TEMPLATE"
                    else int(template_id)
                )
                sequences[block_id].append((int(line_id), numeric_id))

    # 5. Buscar as mensagens originais das linhas exibidas.
    wanted_lines = {
        line_id
        for events in sequences.values()
        for line_id, _ in events[:SHOW_EVENTS]
    }
    originals: dict[int, str] = {}
    with RAW_LOG_PATH.open(encoding="utf-8") as stream:
        for line_id, line in enumerate(stream, start=1):
            if line_id in wanted_lines:
                originals[line_id] = line.rstrip("\r\n")
                if len(originals) == len(wanted_lines):
                    break

    # 6. Exibir para conferência manual.
    for block_id in targets:
        assignment = split_index[block_id]
        events = sequences[block_id]
        ids = [numeric_id for _, numeric_id in events]

        print("=" * 70)
        print(
            f"Sessão {block_id}"
            f" | split={assignment.split}"
            f" | label={assignment.label}"
        )
        print(f"Length={len(events)} | UNK={ids.count(UNK_ID)}")
        shown = " ".join(str(i) for i in ids[:SHOW_SEQUENCE])
        suffix = " ..." if len(ids) > SHOW_SEQUENCE else ""
        print(f"Sequence: {shown}{suffix}")

        print(f"Primeiros {min(SHOW_EVENTS, len(events))} eventos:")
        for line_id, numeric_id in events[:SHOW_EVENTS]:
            print(f"  linha {line_id} -> t{numeric_id}")
            print(f"    template : {templates[numeric_id][:110]}")
            print(f"    original : {originals.get(line_id, '?')[:110]}")


if __name__ == "__main__":
    main()