"""Etapa 4B.2: constrói o corpus textual por sessão."""

import csv
import gzip
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from log_analytics.data.audit_hdfs_v1 import sha256_file

TEMPLATES_PATH = Path(
    "artifacts/parsing/hdfs_v1_templates.csv"
)
PARSING_MANIFEST_PATH = Path(
    "artifacts/parsing/hdfs_v1_parsing_manifest.json"
)
SEQUENCES_PATH = Path(
    "artifacts/representation/hdfs_v1_session_sequences.csv.gz"
)
SEQUENCES_MANIFEST_PATH = Path(
    "artifacts/representation/hdfs_v1_sequences_manifest.json"
)
UNKNOWN_EVENTS_PATH = Path(
    "artifacts/representation/hdfs_v1_unknown_events.csv.gz"
)
UNKNOWN_MANIFEST_PATH = Path(
    "artifacts/representation/hdfs_v1_unknown_events_manifest.json"
)

OUTPUT_DIR = Path("artifacts/representation")
CORPUS_PATH = OUTPUT_DIR / "hdfs_v1_session_corpus.csv.gz"
MANIFEST_PATH = OUTPUT_DIR / "hdfs_v1_session_corpus_manifest.json"

SEQUENCE_HEADER = [
    "BlockId",
    "Split",
    "Length",
    "Sequence",
]
UNKNOWN_HEADER = [
    "LineId",
    "BlockId",
    "Split",
    "PreparedText",
    "TextHash",
]
CORPUS_HEADER = [
    "BlockId",
    "Split",
    "Length",
    "Document",
    "DocumentHash",
]


def load_accepted_manifest(path: Path) -> dict:
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


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_templates(expected_clusters: int) -> dict[int, str]:
    templates: dict[int, str] = {}

    with TEMPLATES_PATH.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        for row in reader:
            template_id = int(row["TemplateId"])
            template = row["Template"]

            if template_id in templates:
                raise ValueError(
                    f"TemplateId duplicado: {template_id}"
                )
            if not template:
                raise ValueError(
                    f"Template vazio: {template_id}"
                )

            templates[template_id] = template

    expected_ids = set(range(1, expected_clusters + 1))
    if set(templates) != expected_ids:
        raise ValueError("Catálogo de templates incompleto.")

    return templates


def load_unknown_events():
    unknown_by_session = defaultdict(deque)
    unknown_by_split: Counter[str] = Counter()
    previous_line_id = 0

    with gzip.open(
        UNKNOWN_EVENTS_PATH,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.reader(stream)

        if next(reader, None) != UNKNOWN_HEADER:
            raise ValueError(
                "Cabeçalho inválido no artefato de UNK."
            )

        for row in reader:
            if len(row) != 5:
                raise ValueError(
                    "Linha inválida no artefato de UNK."
                )

            (
                line_id_text,
                block_id,
                split,
                prepared_text,
                text_hash,
            ) = row

            line_id = int(line_id_text)

            if line_id <= previous_line_id:
                raise ValueError(
                    "LineId de UNK duplicado ou fora de ordem."
                )
            previous_line_id = line_id

            if not prepared_text:
                raise ValueError(
                    f"Texto vazio na linha {line_id}."
                )

            if hash_text(prepared_text) != text_hash:
                raise ValueError(
                    f"Hash inválido na linha {line_id}."
                )

            unknown_by_session[block_id].append(
                (line_id, split, prepared_text)
            )
            unknown_by_split[split] += 1

    return unknown_by_session, unknown_by_split


def main() -> None:
    existing = [
        str(path)
        for path in (CORPUS_PATH, MANIFEST_PATH)
        if path.exists()
    ]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    parsing_manifest = load_accepted_manifest(
        PARSING_MANIFEST_PATH
    )
    sequences_manifest = load_accepted_manifest(
        SEQUENCES_MANIFEST_PATH
    )
    unknown_manifest = load_accepted_manifest(
        UNKNOWN_MANIFEST_PATH
    )

    checks = {
        "templates_sha256_matches_stage3": (
            sha256_file(TEMPLATES_PATH)
            == parsing_manifest["outputs"]["templates_csv"]["sha256"]
        ),
        "sequences_sha256_matches_manifest": (
            sha256_file(SEQUENCES_PATH)
            == sequences_manifest["outputs"][
                "session_sequences_csv_gz"
            ]["sha256"]
        ),
        "unknown_sha256_matches_manifest": (
            sha256_file(UNKNOWN_EVENTS_PATH)
            == unknown_manifest["outputs"][
                "unknown_events_csv_gz"
            ]["sha256"]
        ),
        "common_parsing_provenance": (
            sequences_manifest["inputs"]["parsed_events_sha256"]
            == unknown_manifest["inputs"]["parsed_events_sha256"]
        ),
    }

    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    templates = load_templates(
        parsing_manifest["fit"]["clusters"]
    )
    unknown_by_session, loaded_unknown_by_split = (
        load_unknown_events()
    )

    expected_sessions = sequences_manifest[
        "results"
    ]["sessions_by_split"]
    expected_events = sequences_manifest[
        "results"
    ]["events_by_split"]
    expected_unknown = unknown_manifest[
        "results"
    ]["unknown_by_split"]

    sessions_by_split: Counter[str] = Counter()
    events_by_split: Counter[str] = Counter()
    used_unknown_by_split: Counter[str] = Counter()

    previous_block_id = ""
    documents_nonempty = True
    lengths_valid = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_tmp = Path(f"{CORPUS_PATH}.tmp")

    with corpus_tmp.open("wb") as raw_output:
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
                writer = csv.writer(text_output)
                writer.writerow(CORPUS_HEADER)

                with gzip.open(
                    SEQUENCES_PATH,
                    "rt",
                    encoding="utf-8",
                    newline="",
                ) as sequence_stream:
                    reader = csv.reader(sequence_stream)

                    if next(reader, None) != SEQUENCE_HEADER:
                        raise ValueError(
                            "Cabeçalho inválido nas sequências."
                        )

                    for row in reader:
                        if len(row) != 4:
                            raise ValueError(
                                "Linha de sequência inválida."
                            )

                        (
                            block_id,
                            split,
                            length_text,
                            sequence_text,
                        ) = row

                        if block_id <= previous_block_id:
                            raise ValueError(
                                "BlockId duplicado ou fora de ordem."
                            )
                        previous_block_id = block_id

                        length = int(length_text)
                        template_ids = [
                            int(value)
                            for value in sequence_text.split()
                        ]

                        if len(template_ids) != length:
                            lengths_valid = False

                        event_texts: list[str] = []
                        unknown_queue = unknown_by_session.get(
                            block_id
                        )

                        for template_id in template_ids:
                            if template_id != 0:
                                template = templates.get(template_id)
                                if template is None:
                                    raise ValueError(
                                        f"TemplateId inválido: "
                                        f"{template_id}"
                                    )
                                event_texts.append(template)
                                continue

                            if not unknown_queue:
                                raise ValueError(
                                    f"UNK sem texto para {block_id}."
                                )

                            _, unknown_split, prepared_text = (
                                unknown_queue.popleft()
                            )

                            if unknown_split != split:
                                raise ValueError(
                                    f"Split divergente em {block_id}."
                                )

                            event_texts.append(prepared_text)
                            used_unknown_by_split[split] += 1

                        if unknown_queue:
                            raise ValueError(
                                f"Textos UNK excedentes em {block_id}."
                            )

                        unknown_by_session.pop(block_id, None)

                        document = "\n".join(event_texts)
                        if not document:
                            documents_nonempty = False

                        writer.writerow(
                            [
                                block_id,
                                split,
                                length,
                                document,
                                hash_text(document),
                            ]
                        )

                        sessions_by_split[split] += 1
                        events_by_split[split] += length

    remaining_unknown = sum(
        len(queue)
        for queue in unknown_by_session.values()
    )

    checks.update(
        {
            "loaded_unknown_matches_manifest": (
                dict(loaded_unknown_by_split)
                == expected_unknown
            ),
            "all_unknown_consumed": remaining_unknown == 0,
            "used_unknown_matches_manifest": (
                dict(used_unknown_by_split)
                == expected_unknown
            ),
            "sessions_by_split_preserved": (
                dict(sessions_by_split)
                == expected_sessions
            ),
            "events_by_split_preserved": (
                dict(events_by_split)
                == expected_events
            ),
            "all_lengths_valid": lengths_valid,
            "documents_nonempty": documents_nonempty,
        }
    )

    accepted = all(checks.values())

    if not accepted:
        raise SystemExit(f"Critérios reprovados: {checks}")

    os.replace(corpus_tmp, CORPUS_PATH)

    manifest = {
        "stage": "etapa4b_incremento2_session_corpus",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
        },
        "representation": {
            "unit": "sessão BlockId",
            "event_separator": "LF",
            "known_event_text": "texto do template Drain",
            "unknown_event_text": "PreparedText da condição D",
            "event_repetitions_preserved": True,
            "label_included": False,
        },
        "inputs": {
            "templates_sha256": sha256_file(TEMPLATES_PATH),
            "sequences_sha256": sha256_file(SEQUENCES_PATH),
            "unknown_events_sha256": sha256_file(
                UNKNOWN_EVENTS_PATH
            ),
        },
        "results": {
            "sessions_by_split": dict(sessions_by_split),
            "events_by_split": dict(events_by_split),
            "unknown_by_split": dict(used_unknown_by_split),
        },
        "outputs": {
            "session_corpus_csv_gz": {
                "path": str(CORPUS_PATH),
                "sha256": sha256_file(CORPUS_PATH),
                "size_bytes": CORPUS_PATH.stat().st_size,
            }
        },
        "acceptance": {
            "checks": checks,
            "accepted": accepted,
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
            {
                "accepted": accepted,
                "checks": checks,
                "results": manifest["results"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()