"""Incremento 6 — execução completa: ajuste do Drain em train_normal e
transformação das três partições com o parser congelado (condição C)."""

import csv
import gzip
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from inspect import getfile
from pathlib import Path
from time import perf_counter

import jsonpickle
from drain3.file_persistence import FilePersistence

import log_analytics.data.prepare_hdfs_v1 as prepare_module
from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.fit_hdfs_v1 import (
    build_demo_parser,
    learn_if_train_normal,
)
from log_analytics.data.match_hdfs_v1 import (
    UNK_TEMPLATE,
    match_hdfs_message,
)
from log_analytics.data.prepare_hdfs_v1 import (
    ABSOLUTE_PATH_PATTERN,
    IPV4_BARE_PATTERN,
    IPV4_PORT_PATTERN,
    JOB_ID_PATTERN,
    prepare_hdfs_message_candidate_d,
)
from log_analytics.data.session_hdfs_v1 import (
    associate_log_line,
    load_split_index,
)

RAW_LOG_PATH = Path("data/raw/hdfs_v1/HDFS.log")
SPLIT_CSV_PATH = Path("artifacts/data_splits/hdfs_v1_session_split.csv")
SPLIT_MANIFEST_PATH = Path(
    "artifacts/data_splits/hdfs_v1_split_manifest.json"
)
TRAIN_NORMAL_LIST_PATH = Path(
    "artifacts/data_splits/hdfs_v1_train_normal_block_ids.txt"
)
OUTPUT_DIR = Path("artifacts/parsing")
STATE_PATH = OUTPUT_DIR / "hdfs_v1_drain_state.bin"
TEMPLATES_PATH = OUTPUT_DIR / "hdfs_v1_templates.csv"
EVENTS_PATH = OUTPUT_DIR / "hdfs_v1_parsed_events.csv.gz"
MANIFEST_PATH = OUTPUT_DIR / "hdfs_v1_parsing_manifest.json"

EXPECTED_RAW_LOG_SHA256 = (
    "0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52"
)
EXPECTED_TOTAL_LINES = 11_175_629
EXPECTED_FIT_EVENTS = 7_621_871
EXPECTED_SESSIONS_BY_SPLIT = {
    "train": 402_542,
    "validation": 86_260,
    "test": 86_259,
}
PROBE_EVENTS = 5_000
PROGRESS_EVERY = 250_000

PREPARE = prepare_hdfs_message_candidate_d
PREPARE_NAME = "prepare_hdfs_message_candidate_d"

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def log_progress(
    phase: str,
    line_id: int,
    total: int,
    started: float,
) -> None:
    elapsed = perf_counter() - started
    rate = line_id / elapsed if elapsed > 0 else 0.0
    remaining_minutes = (
        (total - line_id) / rate / 60 if rate > 0 else float("inf")
    )
    log(
        f"{phase}: {line_id:_}/{total:_} linhas"
        f" ({line_id / total:.0%})"
        f" | {rate:_.0f} linhas/s"
        f" | restam ~{remaining_minutes:.1f} min"
    )

def catalog(parser) -> tuple:
    return tuple(
        (c.cluster_id, c.get_template(), c.size)
        for c in sorted(
            parser.drain.clusters,
            key=lambda c: c.cluster_id,
        )
    )


def main() -> None:
    # 0. Recusar sobrescrita: reexecução exige remoção deliberada.
    existing = [
        str(p)
        for p in (STATE_PATH, TEMPLATES_PATH, EVENTS_PATH, MANIFEST_PATH)
        if p.exists()
    ]
    if existing:
        raise SystemExit(f"Artefatos já existem: {existing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validar entradas contra as etapas anteriores.
    log("Fase 1/6 — validando checksums das entradas (o log bruto demora)...")
    raw_log_sha256 = sha256_file(RAW_LOG_PATH)
    split_manifest = json.loads(
        SPLIT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    acceptance = split_manifest["acceptance"]
    split_manifest_ok = acceptance.get("accepted") is True and all(
        value is True for value in acceptance["checks"].values()
    )
    split_csv_sha256 = sha256_file(SPLIT_CSV_PATH)
    train_normal_sha256 = sha256_file(TRAIN_NORMAL_LIST_PATH)

    checks = {
        "raw_log_sha256_matches_stage1": (
            raw_log_sha256 == EXPECTED_RAW_LOG_SHA256
        ),
        "split_manifest_accepted": split_manifest_ok,
        "split_csv_sha256_matches_manifest": (
            split_csv_sha256
            == split_manifest["outputs"]["master_csv"]["sha256"]
        ),
        "train_normal_list_sha256_matches_manifest": (
            train_normal_sha256
            == split_manifest["outputs"]["train_normal"]["sha256"]
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Entradas inválidas: {checks}")

    log("Fase 2/6 — carregando índice do split (575 mil sessões)...")
    split_index = load_split_index(SPLIT_CSV_PATH)

    # 2. Passada 1 — ajuste somente em train_normal, com condição C.
    log("Fase 3/6 — passada 1: ajuste em train_normal com condição C...")
    parser = build_demo_parser(FilePersistence(str(STATE_PATH)))
    probe_messages: list[str] = []
    total_lines = 0
    fit_events = 0
    started = perf_counter()

    with RAW_LOG_PATH.open(encoding="utf-8") as stream:
        for line_id, line in enumerate(stream, start=1):
            total_lines = line_id
            item = associate_log_line(line, line_id, split_index)
            result = learn_if_train_normal(parser, item, PREPARE)

            if result is not None:
                fit_events += 1
                if len(probe_messages) < PROBE_EVENTS:
                    probe_messages.append(item.event.content)

            if line_id % PROGRESS_EVERY == 0:
                log_progress(
                    "ajuste", line_id, EXPECTED_TOTAL_LINES, started
                )

    fit_seconds = perf_counter() - started
    checks["total_lines_expected"] = total_lines == EXPECTED_TOTAL_LINES
    checks["fit_events_expected"] = fit_events == EXPECTED_FIT_EVENTS

    # 3. Congelar: salvar, recarregar e provar a ida e volta.
    log(
        f"Ajuste concluído: {fit_events:_} eventos,"
        f" {len(list(parser.drain.clusters)):_} templates,"
        f" {fit_seconds / 60:.1f} min."
    )

    log("Fase 4/6 — salvando, recarregando e sondando o estado congelado...")
    parser.save_state("fit_complete")
    state_sha256 = sha256_file(STATE_PATH)
    catalog_after_fit = catalog(parser)

    frozen = build_demo_parser(FilePersistence(str(STATE_PATH)))
    checks["state_roundtrip_catalog_identical"] = (
        catalog(frozen) == catalog_after_fit
    )

    probe_memory = [
        match_hdfs_message(parser, message, PREPARE)[0]
        for message in probe_messages
    ]
    probe_frozen = [
        match_hdfs_message(frozen, message, PREPARE)[0]
        for message in probe_messages
    ]
    checks["probe_assignments_identical"] = (
        probe_memory == probe_frozen and None not in probe_frozen
    )

    # 4. Passada 2 — transformar tudo com o parser congelado.
    log("Fase 5/6 — passada 2: transformação das 11,2M de linhas...")
    state_before = jsonpickle.dumps(frozen.drain, keys=True)
    events_by_split: Counter = Counter()
    unk_by_split: Counter = Counter()
    template_matches: Counter = Counter()
    sessions_seen: dict[str, set] = {}
    train_normal_events = 0
    unk_train_normal = 0
    started = perf_counter()

    events_tmp = Path(str(EVENTS_PATH) + ".tmp")
    with events_tmp.open("wb") as raw_stream:
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", mtime=0
        ) as gz_stream:
            with io.TextIOWrapper(
                gz_stream, encoding="utf-8", newline=""
            ) as text_stream:
                writer = csv.writer(text_stream)
                writer.writerow(
                    ["LineId", "BlockId", "Split", "TemplateId"]
                )

                with RAW_LOG_PATH.open(encoding="utf-8") as stream:
                    for line_id, line in enumerate(stream, start=1):
                        item = associate_log_line(
                            line, line_id, split_index
                        )
                        assignment = item.assignment
                        split_key = (
                            assignment.split if assignment else "none"
                        )
                        block_id = (
                            assignment.block_id if assignment else ""
                        )

                        cluster_id, _ = match_hdfs_message(
                            frozen, item.event.content, PREPARE
                        )

                        events_by_split[split_key] += 1
                        if assignment is not None:
                            sessions_seen.setdefault(
                                split_key, set()
                            ).add(block_id)
                        if item.is_train_normal:
                            train_normal_events += 1

                        if cluster_id is None:
                            unk_by_split[split_key] += 1
                            if item.is_train_normal:
                                unk_train_normal += 1
                            template_id = UNK_TEMPLATE
                        else:
                            template_matches[
                                (cluster_id, split_key)
                            ] += 1
                            template_id = cluster_id

                        writer.writerow(
                            [line_id, block_id, split_key, template_id]
                        )

                        if line_id % PROGRESS_EVERY == 0:
                            log_progress(
                                "transformação",
                                line_id,
                                EXPECTED_TOTAL_LINES,
                                started,
                            )

    transform_seconds = perf_counter() - started
    os.replace(events_tmp, EVENTS_PATH)

    checks["transform_total_equals_scan"] = (
        sum(events_by_split.values()) == EXPECTED_TOTAL_LINES
    )
    checks["transform_train_normal_events_expected"] = (
        train_normal_events == EXPECTED_FIT_EVENTS
    )
    checks["no_templates_created_on_transform"] = (
        catalog(frozen) == catalog_after_fit
    )
    checks["state_in_memory_preserved"] = (
        jsonpickle.dumps(frozen.drain, keys=True) == state_before
    )
    checks["state_file_checksum_unchanged"] = (
        sha256_file(STATE_PATH) == state_sha256
    )
    checks["unk_zero_on_train_normal"] = unk_train_normal == 0
    checks["sessions_by_split_expected"] = {
        key: len(value) for key, value in sessions_seen.items()
    } == EXPECTED_SESSIONS_BY_SPLIT

    # 5. Catálogo de templates com contagens por partição.
    log(
        f"Transformação concluída em {transform_seconds / 60:.1f} min."
        f" UNK por partição: {dict(unk_by_split)}."
    )

    log("Fase 6/6 — gravando templates, manifesto e checksums finais...")
    templates_tmp = Path(str(TEMPLATES_PATH) + ".tmp")
    with templates_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "TemplateId", "FitCount",
            "MatchTrain", "MatchValidation", "MatchTest", "MatchNone",
            "Template",
        ])
        for cluster_id, template, size in catalog_after_fit:
            writer.writerow([
                cluster_id,
                size,
                template_matches[(cluster_id, "train")],
                template_matches[(cluster_id, "validation")],
                template_matches[(cluster_id, "test")],
                template_matches[(cluster_id, "none")],
                template,
            ])
    os.replace(templates_tmp, TEMPLATES_PATH)

    # 6. Manifesto.
    accepted = all(checks.values())
    manifest = {
        "stage": "etapa3_incremento6_parsing_completo",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "drain3": version("drain3"),
        },
        "parser_config": {
            "drain_depth": frozen.config.drain_depth,
            "drain_sim_th": frozen.config.drain_sim_th,
            "drain_max_children": frozen.config.drain_max_children,
            "drain_max_clusters": frozen.config.drain_max_clusters,
            "parametrize_numeric_tokens": (
                frozen.config.parametrize_numeric_tokens
            ),
            "extra_delimiters": frozen.config.drain_extra_delimiters,
            "masking_instructions": len(
                frozen.config.masking_instructions
            ),
            "full_search_strategy": "always",
        },
        "preparation": {
            "name": PREPARE_NAME,
            "module_sha256": sha256_file(Path(getfile(prepare_module))),
            "ipv4_port_pattern": IPV4_PORT_PATTERN.pattern,
            "job_id_pattern": JOB_ID_PATTERN.pattern,
            "ipv4_bare_pattern": IPV4_BARE_PATTERN.pattern,
            "path_pattern": ABSOLUTE_PATH_PATTERN.pattern,
        },
        "inputs": {
            "raw_log_sha256": raw_log_sha256,
            "split_csv_sha256": split_csv_sha256,
            "train_normal_list_sha256": train_normal_sha256,
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
        },
        "fit": {
            "events": fit_events,
            "clusters": len(catalog_after_fit),
            "singleton_clusters": sum(
                size == 1 for _, _, size in catalog_after_fit
            ),
            "probe_events": len(probe_messages),
            "seconds": fit_seconds,
        },
        "transform": {
            "total_lines": total_lines,
            "events_by_split": dict(events_by_split),
            "train_normal_events": train_normal_events,
            "unk_by_split": dict(unk_by_split),
            "unk_train_normal": unk_train_normal,
            "sessions_by_split": {
                key: len(value)
                for key, value in sorted(sessions_seen.items())
            },
            "seconds": transform_seconds,
        },
        "outputs": {
            "drain_state": {
                "path": str(STATE_PATH),
                "sha256": state_sha256,
                "size_bytes": STATE_PATH.stat().st_size,
            },
            "templates_csv": {
                "path": str(TEMPLATES_PATH),
                "sha256": sha256_file(TEMPLATES_PATH),
                "size_bytes": TEMPLATES_PATH.stat().st_size,
            },
            "parsed_events_csv_gz": {
                "path": str(EVENTS_PATH),
                "sha256": sha256_file(EVENTS_PATH),
                "size_bytes": EVENTS_PATH.stat().st_size,
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
        {
            "accepted": accepted,
            "checks": checks,
            "fit": manifest["fit"],
            "transform": manifest["transform"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
