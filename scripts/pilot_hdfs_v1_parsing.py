import inspect
import json
from collections import Counter
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from time import perf_counter

import jsonpickle

from log_analytics.data.audit_hdfs_v1 import sha256_file
from log_analytics.data.fit_hdfs_v1 import (
    build_demo_parser,
    learn_if_train_normal,
)
from log_analytics.data.match_hdfs_v1 import match_hdfs_message
from log_analytics.data.prepare_hdfs_v1 import prepare_hdfs_message
from log_analytics.data.session_hdfs_v1 import (
    associate_log_line,
    load_split_index,
)


TARGET_EVENTS = 5_000
MAX_LINES = 100_000


def main() -> None:
    # 1. Conferir a referência da divisão.
    split_path = Path("artifacts/data_splits/hdfs_v1_session_split.csv")
    manifest_path = Path(
        "artifacts/data_splits/hdfs_v1_split_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    acceptance = manifest["acceptance"]
    checks = acceptance.get("checks", {})
    if acceptance.get("accepted") is not True or not checks or any(
        value is not True for value in checks.values()
    ):
        raise ValueError("Manifesto da etapa 2 não aprovado.")

    split_sha256 = sha256_file(split_path)
    if split_sha256 != manifest["outputs"]["master_csv"]["sha256"]:
        raise ValueError("CSV mestre diverge do manifesto.")

    split_index = load_split_index(split_path)

    # 2. Selecionar somente eventos normais do treino.
    sample = []
    skipped = Counter()
    lines_read = 0
    started = perf_counter()

    with Path("data/raw/hdfs_v1/HDFS.log").open(
        encoding="utf-8"
    ) as stream:
        for line_id, line in enumerate(
            islice(stream, MAX_LINES), start=1
        ):
            lines_read = line_id
            item = associate_log_line(line, line_id, split_index)

            if item.is_train_normal:
                sample.append(item)
                if len(sample) == TARGET_EVENTS:
                    break
            else:
                assignment = item.assignment
                reason = (
                    f"{assignment.split}/{assignment.label}"
                    if assignment else "sem_sessao"
                )
                skipped[reason] += 1

    selection_seconds = perf_counter() - started

    if not sample:
        raise ValueError("Nenhum evento elegível na faixa percorrida.")

    # 3. Ajustar um parser novo usando a proteção do incremento 4.
    parser = build_demo_parser()
    started = perf_counter()

    for item in sample:
        result = learn_if_train_normal(parser, item)
        assert result is not None

    fit_seconds = perf_counter() - started

    # 4. Reconhecer a mesma amostra com o estado congelado.
    before = jsonpickle.dumps(parser.drain, keys=True)
    examples = {}
    unknown = []
    started = perf_counter()

    for item in sample:
        cluster_id, template = match_hdfs_message(
            parser, item.event.content
        )
        example = {
            "line_id": item.event.line_id,
            "component": item.event.component,
            "original": item.event.content,
            "prepared": prepare_hdfs_message(item.event.content),
            "template": template,
        }

        if cluster_id is None:
            unknown.append(example)
        else:
            examples.setdefault(cluster_id, [])
            if len(examples[cluster_id]) < 2:
                examples[cluster_id].append(example)

    match_seconds = perf_counter() - started
    state_preserved = (
        jsonpickle.dumps(parser.drain, keys=True) == before
    )

    # 5. Separar exemplos para revisão.
    clusters = list(parser.drain.clusters)

    frequent = sorted(
        clusters, key=lambda c: (-c.size, c.cluster_id)
    )[:3]
    rare = sorted(
        clusters, key=lambda c: (c.size, c.cluster_id)
    )[:3]
    general = sorted(
        clusters,
        key=lambda c: (
            -c.get_template().count("<*>"),
            c.cluster_id,
        ),
    )[:3]

    selected = {
        c.cluster_id: c
        for c in frequent + rare + general
    }

    # 6. Produzir o relatório do piloto.
    report = {
        "scope": "piloto no próprio treino; não mede generalização",
        "drain3_version": version("drain3"),
        "config": vars(parser.config),
        "preparer_sha256": sha256_file(
            Path(inspect.getfile(prepare_hdfs_message))
        ),
        "split_sha256": split_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "raw_log_checksum_revalidated": False,
        "lines_read": lines_read,
        "selected_events": len(sample),
        "target_reached": len(sample) == TARGET_EVENTS,
        "stop_reason": (
            "target_events" if len(sample) == TARGET_EVENTS
            else "line_limit" if lines_read == MAX_LINES
            else "end_of_file"
        ),
        "excluded_events": dict(skipped),
        "selected_sessions": len({
            item.assignment.block_id for item in sample
        }),
        "components": dict(Counter(
            item.event.component for item in sample
        )),
        "severities": dict(Counter(
            item.event.level for item in sample
        )),
        "clusters": len(clusters),
        "singleton_clusters": sum(
            c.size == 1 for c in clusters
        ),
        "fit_event_count": sum(c.size for c in clusters),
        "unknown_on_fit_sample": len(unknown),
        "unknown_examples": unknown[:3],
        "state_preserved": state_preserved,
        "seconds": {
            "selection": selection_seconds,
            "fit": fit_seconds,
            "rematch": match_seconds,
        },
        "catalog_examples": [
            {
                "cluster_id": c.cluster_id,
                "fit_count": c.size,
                "template": c.get_template(),
                "examples": examples.get(c.cluster_id, []),
            }
            for c in selected.values()
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert lines_read == len(sample) + sum(skipped.values())
    assert report["fit_event_count"] == len(sample)
    assert state_preserved


if __name__ == "__main__":
    main()