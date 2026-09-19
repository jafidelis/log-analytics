"""Etapa 4B.1: preserva o texto preparado dos eventos UNK_TEMPLATE."""

import csv
import gzip
import hashlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from inspect import getfile
from itertools import zip_longest
from pathlib import Path

import log_analytics.data.prepare_hdfs_v1 as prepare_module
from log_analytics.data.audit_hdfs_v1 import (
    extract_block_ids,
    parse_log_line,
    sha256_file,
)
from log_analytics.data.match_hdfs_v1 import UNK_TEMPLATE
from log_analytics.data.prepare_hdfs_v1 import (
    prepare_hdfs_message_candidate_d,
)
from log_analytics.data.session_hdfs_v1 import load_split_index
from log_analytics.data.split_hdfs_v1 import (
    LABEL_NORMAL,
    SPLIT_TRAIN,
)

RAW_LOG_PATH = Path("data/raw/hdfs_v1/HDFS.log")
PARSED_EVENTS_PATH = Path(
    "artifacts/parsing/hdfs_v1_parsed_events.csv.gz"
)
PARSING_MANIFEST_PATH = Path(
    "artifacts/parsing/hdfs_v1_parsing_manifest.json"
)
SPLIT_CSV_PATH = Path(
    "artifacts/data_splits/hdfs_v1_session_split.csv"
)

OUTPUT_DIR = Path("artifacts/representation")
UNKNOWN_EVENTS_PATH = (
    OUTPUT_DIR / "hdfs_v1_unknown_events.csv.gz"
)
MANIFEST_PATH = (
    OUTPUT_DIR / "hdfs_v1_unknown_events_manifest.json"
)

PREPARE = prepare_hdfs_message_candidate_d
PREPARE_NAME = "prepare_hdfs_message_candidate_d"

PARSED_HEADER = [
    "LineId",
    "BlockId",
    "Split",
    "TemplateId",
]

OUTPUT_HEADER = [
    "LineId",
    "BlockId",
    "Split",
    "PreparedText",
    "TextHash",
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
    """Calcula SHA-256 sobre o texto preparado em UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    existing = [
        str(path)
        for path in (UNKNOWN_EVENTS_PATH, MANIFEST_PATH)
        if path.exists()
    ]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    parsing_manifest = load_accepted_manifest(
        PARSING_MANIFEST_PATH
    )

    raw_log_sha256 = sha256_file(RAW_LOG_PATH)
    parsed_events_sha256 = sha256_file(PARSED_EVENTS_PATH)
    split_csv_sha256 = sha256_file(SPLIT_CSV_PATH)
    preparation_sha256 = sha256_file(
        Path(getfile(prepare_module))
    )

    checks = {
        "raw_log_sha256_matches_stage3": (
            raw_log_sha256
            == parsing_manifest["inputs"]["raw_log_sha256"]
        ),
        "parsed_events_sha256_matches_stage3": (
            parsed_events_sha256
            == parsing_manifest["outputs"][
                "parsed_events_csv_gz"
            ]["sha256"]
        ),
        "split_csv_sha256_matches_stage3": (
            split_csv_sha256
            == parsing_manifest["inputs"]["split_csv_sha256"]
        ),
        "preparation_sha256_matches_stage3": (
            preparation_sha256
            == parsing_manifest["preparation"]["module_sha256"]
        ),
        "preparation_name_matches_stage3": (
            parsing_manifest["preparation"]["name"]
            == PREPARE_NAME
        ),
    }

    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    split_index = load_split_index(SPLIT_CSV_PATH)

    expected_total_lines = parsing_manifest[
        "transform"
    ]["total_lines"]
    expected_unk_by_split = parsing_manifest[
        "transform"
    ]["unk_by_split"]
    expected_unk_train_normal = parsing_manifest[
        "transform"
    ]["unk_train_normal"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_tmp = Path(f"{UNKNOWN_EVENTS_PATH}.tmp")

    total_lines = 0
    unknown_by_split: Counter[str] = Counter()
    unknown_train_normal = 0
    empty_prepared_text = 0

    with output_tmp.open("wb") as raw_output:
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
                writer.writerow(OUTPUT_HEADER)

                with RAW_LOG_PATH.open(
                    encoding="utf-8"
                ) as raw_stream:
                    with gzip.open(
                        PARSED_EVENTS_PATH,
                        "rt",
                        encoding="utf-8",
                        newline="",
                    ) as parsed_stream:
                        reader = csv.reader(parsed_stream)

                        header = next(reader, None)
                        if header != PARSED_HEADER:
                            raise ValueError(
                                f"Cabeçalho inesperado: {header}"
                            )

                        pairs = zip_longest(raw_stream, reader)

                        for expected_line_id, pair in enumerate(
                            pairs,
                            start=1,
                        ):
                            raw_line, parsed_row = pair

                            if raw_line is None or parsed_row is None:
                                raise ValueError(
                                    "Log bruto e parsing possuem "
                                    "quantidades diferentes de linhas."
                                )

                            if len(parsed_row) != 4:
                                raise ValueError(
                                    f"Linha {expected_line_id}: "
                                    "esperados quatro campos."
                                )

                            (
                                line_id_text,
                                block_id,
                                split,
                                template_id,
                            ) = parsed_row

                            line_id = int(line_id_text)
                            if line_id != expected_line_id:
                                raise ValueError(
                                    f"LineId fora de ordem: "
                                    f"{line_id} != {expected_line_id}"
                                )

                            total_lines = expected_line_id

                            if template_id != UNK_TEMPLATE:
                                continue

                            event = parse_log_line(
                                raw_line.rstrip("\r\n"),
                                line_id,
                            )
                            if event is None:
                                raise ValueError(
                                    f"Linha {line_id}: "
                                    "cabeçalho inválido."
                                )

                            block_ids = extract_block_ids(
                                event.content
                            )
                            if tuple(block_ids) != (block_id,):
                                raise ValueError(
                                    f"Linha {line_id}: BlockId "
                                    "diverge do parsing."
                                )

                            assignment = split_index.get(block_id)
                            if assignment is None:
                                raise ValueError(
                                    f"Linha {line_id}: sessão "
                                    "ausente no split."
                                )

                            if assignment.split != split:
                                raise ValueError(
                                    f"Linha {line_id}: partição "
                                    "diverge do split mestre."
                                )

                            prepared_text = PREPARE(event.content)
                            if not prepared_text:
                                empty_prepared_text += 1

                            writer.writerow(
                                [
                                    line_id,
                                    block_id,
                                    split,
                                    prepared_text,
                                    hash_text(prepared_text),
                                ]
                            )

                            unknown_by_split[split] += 1

                            if (
                                assignment.split == SPLIT_TRAIN
                                and assignment.label == LABEL_NORMAL
                            ):
                                unknown_train_normal += 1

    os.replace(output_tmp, UNKNOWN_EVENTS_PATH)

    checks.update(
        {
            "total_lines_match_stage3": (
                total_lines == expected_total_lines
            ),
            "unknown_by_split_match_stage3": (
                dict(unknown_by_split)
                == expected_unk_by_split
            ),
            "unknown_train_normal_match_stage3": (
                unknown_train_normal
                == expected_unk_train_normal
            ),
            "prepared_text_nonempty": (
                empty_prepared_text == 0
            ),
            "unknown_total_expected": (
                sum(unknown_by_split.values()) == 5_484
            ),
        }
    )

    accepted = all(checks.values())

    manifest = {
        "stage": "etapa4b_incremento1_unknown_events",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
        },
        "representation": {
            "unit": "evento UNK_TEMPLATE",
            "text": "mensagem preparada pela condição D",
            "raw_text_included": False,
            "label_included": False,
            "text_hash": "SHA-256 de PreparedText em UTF-8",
        },
        "preparation": {
            "name": PREPARE_NAME,
            "module_sha256": preparation_sha256,
        },
        "inputs": {
            "raw_log_sha256": raw_log_sha256,
            "parsed_events_sha256": parsed_events_sha256,
            "split_csv_sha256": split_csv_sha256,
            "parsing_manifest_sha256": sha256_file(
                PARSING_MANIFEST_PATH
            ),
        },
        "results": {
            "total_lines_scanned": total_lines,
            "unknown_events": sum(
                unknown_by_split.values()
            ),
            "unknown_by_split": dict(unknown_by_split),
            "unknown_train_normal": unknown_train_normal,
            "empty_prepared_text": empty_prepared_text,
        },
        "outputs": {
            "unknown_events_csv_gz": {
                "path": str(UNKNOWN_EVENTS_PATH),
                "sha256": sha256_file(UNKNOWN_EVENTS_PATH),
                "size_bytes": UNKNOWN_EVENTS_PATH.stat().st_size,
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

    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()